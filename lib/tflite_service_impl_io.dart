// lib/tflite_service_impl_io.dart
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteServiceImplIO {
  final String modelAssetPath;
  final String labelsAssetPath;

  Interpreter? _interpreter;
  List<String>? _labels;
  String? lastLoadError;

  TFLiteServiceImplIO({
    required this.modelAssetPath,
    required this.labelsAssetPath,
  });

  List<String> _assetKeyCandidates(String key) {
    final cleaned = key.trim();
    final basename = cleaned.split('/').last;
    final set = <String>{
      cleaned,
      if (!cleaned.startsWith('assets/')) 'assets/$cleaned',
      basename,
      'assets/$basename',
    };
    return set.where((e) => e.isNotEmpty).toList();
  }

  /// Load TFLite model and labels
  Future<bool> loadModel() async {
    lastLoadError = null;
    try {
      if (_interpreter != null) return true;

      final loadErrors = <String>[];

      // Load labels from first valid key.
      String? rawLabels;
      for (final key in _assetKeyCandidates(labelsAssetPath)) {
        try {
          rawLabels = await rootBundle.loadString(key);
          break;
        } catch (e) {
          loadErrors.add('labels:$key => $e');
        }
      }
      if (rawLabels == null) {
        lastLoadError = 'Could not load labels. ${loadErrors.join(' | ')}';
        return false;
      }

      _labels = rawLabels
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      final options = InterpreterOptions()..threads = 2;

      // Try fromAsset first, then ByteData->fromBuffer for each candidate key.
      for (final key in _assetKeyCandidates(modelAssetPath)) {
        try {
          _interpreter = await Interpreter.fromAsset(key, options: options);
          break;
        } catch (e) {
          loadErrors.add('fromAsset:$key => $e');
        }

        try {
          final modelData = await rootBundle.load(key);
          final modelBytes = modelData.buffer.asUint8List(
            modelData.offsetInBytes,
            modelData.lengthInBytes,
          );
          _interpreter = Interpreter.fromBuffer(modelBytes, options: options);
          break;
        } catch (e) {
          loadErrors.add('fromBuffer:$key => $e');
        }
      }

      if (_interpreter == null) {
        lastLoadError = 'Could not load model. ${loadErrors.join(' | ')}';
        return false;
      }

      if (_labels == null || _labels!.isEmpty) {
        lastLoadError = 'Labels file is empty: $labelsAssetPath';
        _interpreter?.close();
        _interpreter = null;
        return false;
      }

      return true;
    } catch (e) {
      lastLoadError = e.toString();
      _interpreter?.close();
      _interpreter = null;
      return false;
    }
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  List<List<List<List<double>>>> _buildInput(img.Image resized,
      {required bool normalizeToUnit}) {
    return List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resized.getPixel(x, y);
            final r = img.getRed(pixel).toDouble();
            final g = img.getGreen(pixel).toDouble();
            final b = img.getBlue(pixel).toDouble();
            if (normalizeToUnit) {
              return [r / 255.0, g / 255.0, b / 255.0];
            }
            return [r, g, b];
          },
        ),
      ),
    );
  }

  List<double> _toProbabilities(List<double> values) {
    final allInZeroOne = values.every((v) => v >= 0.0 && v <= 1.0);
    final sum = values.fold<double>(0.0, (a, b) => a + b);
    if (allInZeroOne && (sum - 1.0).abs() < 0.05) {
      return values;
    }

    final maxV = values.reduce(math.max);
    final exps = values.map((v) => math.exp(v - maxV)).toList();
    final denom = exps.fold<double>(0.0, (a, b) => a + b);
    if (denom == 0) {
      return List<double>.filled(values.length, 1.0 / values.length);
    }
    return exps.map((e) => e / denom).toList();
  }

  List<double> _runAndReadScores(
      List<List<List<List<double>>>> input, int classCount) {
    final output =
        List.generate(1, (_) => List<double>.filled(classCount, 0.0));
    _interpreter!.run(input, output);
    return List<double>.from(output[0]);
  }

  /// Run inference and return label + confidence
  Future<Map<String, dynamic>> runInferenceDebug(File imageFile) async {
    if (_interpreter == null) {
      return {'error': 'Model not loaded'};
    }

    try {
      // 1️⃣ Decode image and preprocess
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) {
        return {'error': 'Invalid image file'};
      }

      // Resize to model input size 224x224.
      final resized = img.copyResize(decoded, width: 224, height: 224);

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final classCount =
          outputShape.isNotEmpty ? outputShape.last : (_labels?.length ?? 1);

      // Try both common EfficientNet input conventions:
      // A) float [0,1], B) float [0,255]. Choose the one with higher top confidence.
      final scoresUnit = _runAndReadScores(
        _buildInput(resized, normalizeToUnit: true),
        classCount,
      );
      final probsUnit = _toProbabilities(scoresUnit);
      final bestUnit = probsUnit.reduce(math.max);

      final scores255 = _runAndReadScores(
        _buildInput(resized, normalizeToUnit: false),
        classCount,
      );
      final probs255 = _toProbabilities(scores255);
      final best255 = probs255.reduce(math.max);

      final usingUnit = bestUnit >= best255;
      final scores = usingUnit ? probsUnit : probs255;

      final maxIdx =
          scores.indexWhere((v) => v == scores.reduce((a, b) => a > b ? a : b));
      final label = _labels != null && maxIdx < _labels!.length
          ? _labels![maxIdx]
          : 'Unknown';
      final confidence = scores[maxIdx];

      // Sort results for debug
      final results = List.generate(scores.length, (i) {
        return {
          'label':
              _labels != null && i < _labels!.length ? _labels![i] : 'Class $i',
          'score': scores[i],
        };
      });
      results.sort(
          (a, b) => (b['score'] as double).compareTo(a['score'] as double));

      return {
        'results': results,
        'topLabel': label,
        'confidence': confidence,
        'preprocess': usingUnit ? 'zero_to_one' : 'zero_to_255',
        'outputShape': outputShape,
      };
    } catch (e) {
      return {'error': e.toString()};
    }
  }
}
