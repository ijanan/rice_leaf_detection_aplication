// lib/tflite_service_impl_io.dart
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteServiceImplIO {
  final String modelAssetPath;
  final String labelsAssetPath;

  Interpreter? _interpreter;
  List<String>? _labels;
  String? lastLoadError;
  int _inputWidth = 224;
  int _inputHeight = 224;
  int _numClasses = 0;

  TFLiteServiceImplIO({
    required this.modelAssetPath,
    required this.labelsAssetPath,
  });

  /// Load TFLite model and labels
  Future<bool> loadModel() async {
    lastLoadError = null;
    try {
      if (_interpreter != null) return true;

      final loadErrors = <String>[];

      // Prefer direct asset loading first; fallback to bytes loading.
      try {
        _interpreter = await Interpreter.fromAsset(modelAssetPath);
      } catch (e) {
        loadErrors.add('fromAsset failed: $e');
      }

      if (_interpreter == null) {
        try {
          final modelData = await rootBundle.load(modelAssetPath);
          _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());
        } catch (e) {
          loadErrors.add('fromBuffer failed: $e');
        }
      }

      if (_interpreter == null) {
        lastLoadError =
            'Could not create interpreter for $modelAssetPath. ${loadErrors.join(' | ')}';
        return false;
      }

      // Read model tensor metadata so preprocessing/output always match this model.
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      if (inputShape.length != 4 || inputShape[3] != 3) {
        lastLoadError = 'Unsupported input tensor shape: $inputShape';
        _interpreter!.close();
        _interpreter = null;
        return false;
      }
      if (outputShape.length < 2 || outputShape.last <= 0) {
        lastLoadError = 'Unsupported output tensor shape: $outputShape';
        _interpreter!.close();
        _interpreter = null;
        return false;
      }
      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];
      _numClasses = outputShape.last;

      // Load labels
      final raw = await rootBundle.loadString(labelsAssetPath);
      _labels = raw
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      if ((_labels?.isEmpty ?? true)) {
        lastLoadError = 'Labels file is empty: $labelsAssetPath';
        _interpreter!.close();
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

  /// Run inference and return label + confidence
  Future<Map<String, dynamic>> runInferenceDebug(File imageFile) async {
    if (_interpreter == null) {
      return {
        'error': 'Model not loaded',
        'errors': ['Interpreter is null. Call loadModel() first.'],
      };
    }

    try {
      // 1️⃣ Decode image and preprocess
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) {
        return {
          'error': 'Invalid image file',
          'errors': ['Could not decode selected image file.'],
        };
      }

      // Resize to model input tensor size.
      final resized =
          img.copyResize(decoded, width: _inputWidth, height: _inputHeight);

      // Normalize to [0,1]
      final input = List.generate(
        1,
        (_) => List.generate(
          _inputHeight,
          (y) => List.generate(
            _inputWidth,
            (x) {
              final pixel = resized.getPixel(x, y);
              final r = img.getRed(pixel) / 255.0;
              final g = img.getGreen(pixel) / 255.0;
              final b = img.getBlue(pixel) / 255.0;
              return [r, g, b];
            },
          ),
        ),
      );

      // 2️⃣ Prepare output buffer
      final outputClassCount =
          _numClasses > 0 ? _numClasses : (_labels?.length ?? 1);
      final output = [List<double>.filled(outputClassCount, 0.0)];

      // 3️⃣ Run inference
      _interpreter!.run(input, output);

      // 4️⃣ Process results
      final scores = List<double>.from(output[0]);
      if (scores.isEmpty) {
        return {
          'error': 'Model output is empty',
          'errors': ['Output tensor has no scores.'],
        };
      }
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
      };
    } catch (e) {
      return {
        'error': e.toString(),
        'errors': ['Inference exception: $e'],
      };
    }
  }
}
