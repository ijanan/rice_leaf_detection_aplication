// lib/tflite_service.dart
import 'dart:io';

abstract class TFLiteService {
  String? lastLoadError;
  bool get modelLoaded;

  Future<bool> loadModel({bool useGpuDelegate = false});

  // For simplicity the interface expects a File or bytes
  Future<Map<String, dynamic>> runInferenceDebug(File imageFile, {int topK = 3});

  void close();
}
