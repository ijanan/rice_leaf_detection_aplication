// lib/tflite_service_impl_io.dart
import 'dart:io';
import 'dart:typed_data';
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

  /// Load TFLite model and labels
  Future<bool> loadModel() async {
    try {
      if (_interpreter != null) return true;

      // Load model
      final modelData = await rootBundle.load(modelAssetPath);
      _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());

      // Load labels
      final raw = await rootBundle.loadString(labelsAssetPath);
      _labels = raw
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      return true;
    } catch (e) {
      lastLoadError = e.toString();
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
      return {'error': 'Model not loaded'};
    }

    try {
      // 1️⃣ Decode image and preprocess
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) {
        return {'error': 'Invalid image file'};
      }

      // Resize to model input size 224x224
      final resized = img.copyResize(decoded, width: 224, height: 224);

      // Normalize to [0,1]
      final input = List.generate(
        1,
        (_) => List.generate(
          224,
          (y) => List.generate(
            224,
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
      final output = List.filled(1, List.filled(_labels?.length ?? 4, 0.0));

      // 3️⃣ Run inference
      _interpreter!.run(input, output);

      // 4️⃣ Process results
      final scores = List<double>.from(output[0]);
      final maxIdx = scores.indexWhere(
          (v) => v == scores.reduce((a, b) => a > b ? a : b));
      final label = _labels != null && maxIdx < _labels!.length
          ? _labels![maxIdx]
          : 'Unknown';
      final confidence = scores[maxIdx];

      // Sort results for debug
      final results = List.generate(scores.length, (i) {
        return {
          'label': _labels != null && i < _labels!.length
              ? _labels![i]
              : 'Class $i',
          'score': scores[i],
        };
      });
      results.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));

      return {
        'results': results,
        'topLabel': label,
        'confidence': confidence,
      };
    } catch (e) {
      return {'error': e.toString()};
    }
  }
}
