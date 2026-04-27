// lib/main.dart
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

import 'tflite_service_impl_io.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final baseText =
        GoogleFonts.montserratTextTheme(Theme.of(context).textTheme);
    final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF2E7D32));
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Leaf Disease Detector',
      theme: ThemeData(
        colorScheme: scheme,
        useMaterial3: true,
        textTheme: baseText,
        appBarTheme: AppBarTheme(
          centerTitle: true,
          backgroundColor: Colors.transparent,
          elevation: 0,
          foregroundColor: scheme.onPrimary,
        ),
        floatingActionButtonTheme: FloatingActionButtonThemeData(
          backgroundColor: scheme.primary,
          foregroundColor: scheme.onPrimary,
          extendedTextStyle:
              baseText.titleMedium?.copyWith(fontWeight: FontWeight.w600),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: scheme.primary,
            foregroundColor: scheme.onPrimary,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            textStyle:
                baseText.labelLarge?.copyWith(fontWeight: FontWeight.w600),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            textStyle:
                baseText.labelLarge?.copyWith(fontWeight: FontWeight.w600),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            side: BorderSide(color: scheme.outlineVariant),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            textStyle:
                baseText.labelLarge?.copyWith(fontWeight: FontWeight.w600),
          ),
        ),
        cardTheme: const CardThemeData(
          elevation: 0,
          // Use Material3 defaults with soft rounded corners
          // surfaceTintColor gets applied automatically from ColorScheme in M3
          // shape is set via CardThemeData in newer Flutter versions
          // Keeping it minimal here for compatibility
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ValueNotifier<Uint8List?> _imageBytes = ValueNotifier(null);
  final ValueNotifier<String?> _result = ValueNotifier(null);
  final ValueNotifier<double?> _confidence = ValueNotifier(null);
  final ValueNotifier<Map<String, dynamic>?> _lastDebug = ValueNotifier(null);
  final ValueNotifier<bool> _isLoading = ValueNotifier(false);
  final ValueNotifier<bool> _modelLoaded = ValueNotifier(false);
  final ValueNotifier<String?> _modelError = ValueNotifier(null);

  final TFLiteServiceImplIO _tfliteService = TFLiteServiceImplIO(
    modelAssetPath: 'assets/efficientnetb0.tflite',
    labelsAssetPath: 'assets/labels.txt',
  );

  @override
  void initState() {
    super.initState();
    _loadModelAsync();
  }

  Future<void> _loadModelAsync() async {
    final ok = await _tfliteService.loadModel();
    if (mounted) {
      _modelLoaded.value = ok == true;
      if (ok == true) {
        _modelError.value = null;
        debugPrint('Model loaded successfully');
      }
      if (!ok && _tfliteService.lastLoadError != null) {
        _modelError.value = _tfliteService.lastLoadError;
        debugPrint('Model load failed: ${_tfliteService.lastLoadError}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Model load failed: ${_tfliteService.lastLoadError}'),
            duration: const Duration(seconds: 5),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _imageBytes.dispose();
    _result.dispose();
    _confidence.dispose();
    _isLoading.dispose();
    _lastDebug.dispose();
    _modelLoaded.dispose();
    _modelError.dispose();
    _tfliteService.close();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final allowed = await _ensurePermissions(source);
      if (!allowed) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Permission denied. Cannot access camera/gallery.'),
        ));
        return;
      }

      if (!_modelLoaded.value) {
        _isLoading.value = true;
        final ok = await _tfliteService.loadModel();
        _modelLoaded.value = ok == true;
        _modelError.value = ok == true ? null : _tfliteService.lastLoadError;
        _isLoading.value = false;
        if (!ok) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: Text(
                  'Model failed to load: ${_tfliteService.lastLoadError ?? "Unknown error"}'),
              backgroundColor: Colors.red,
            ));
          }
          return;
        }
      }

      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        preferredCameraDevice: CameraDevice.rear,
      );
      if (pickedFile == null) return;

      final bytes = await pickedFile.readAsBytes();

      _imageBytes.value = null;
      _result.value = null;
      _confidence.value = null;
      _lastDebug.value = null;

      // Always show the chosen image
      _imageBytes.value = bytes;

      _isLoading.value = true;
      await _runInference(bytes);
    } catch (e) {
      _result.value = 'Error: $e';
      _confidence.value = 0.0;
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Unexpected error: $e'),
          backgroundColor: Colors.red,
        ));
      }
    } finally {
      _isLoading.value = false;
    }
  }

  Future<void> _runInference(Uint8List bytes) async {
    try {
      final tempFile = await _bytesToTempFile(bytes);
      final debug = await _tfliteService.runInferenceDebug(tempFile);
      _lastDebug.value = debug;

      if (debug['results'] != null) {
        final top = (debug['results'] as List).cast<Map>();
        if (top.isNotEmpty) {
          final first = top.first;
          final label = first['label'] ?? 'Unknown';
          final score = (first['score'] as num?)?.toDouble() ?? 0.0;
          if (score < 0.3) {
            _result.value = 'Uncertain ($label)';
          } else {
            _result.value = label.toString();
          }
          _confidence.value = score;
        } else {
          _result.value = 'No valid predictions';
          _confidence.value = 0.0;
        }
      } else {
        _result.value = debug['error'] ?? 'Could not classify image.';
        _confidence.value = 0.0;
      }
    } catch (e) {
      _result.value = 'Error during inference: $e';
      _confidence.value = 0.0;
    }
  }

  Future<File> _bytesToTempFile(Uint8List bytes) async {
    final dir = Directory.systemTemp;
    final file = File('${dir.path}/leaf_temp.jpg');
    await file.writeAsBytes(bytes);
    return file;
  }

  Future<bool> _ensurePermissions(ImageSource source) async {
    if (kIsWeb) return true;
    try {
      if (source == ImageSource.camera) {
        final status = await Permission.camera.request();
        if (status.isPermanentlyDenied) {
          openAppSettings();
          return false;
        }
        return status.isGranted;
      } else {
        final photos = await Permission.photos.request();
        if (photos.isGranted) return true;
        if (photos.isPermanentlyDenied) {
          openAppSettings();
          return false;
        }
        final storage = await Permission.storage.request();
        if (storage.isPermanentlyDenied) {
          openAppSettings();
          return false;
        }
        return storage.isGranted;
      }
    } catch (_) {
      return false;
    }
  }

  // Test button removed per request

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: const Text('Leaf Disease Detector'),
      ),
      body: Stack(
        children: [
          // Gradient header background
          Container(
            height: 240,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  scheme.primary,
                  scheme.tertiary,
                ],
              ),
            ),
          ),
          // Content
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Header text + model status
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Diagnose your leaf',
                                style: textTheme.headlineSmall?.copyWith(
                                  color: scheme.onPrimary,
                                  fontWeight: FontWeight.w700,
                                )),
                            const SizedBox(height: 4),
                            Text('Pick or capture a clear photo',
                                style: textTheme.bodyMedium?.copyWith(
                                  color: scheme.onPrimary.withAlpha(220),
                                )),
                          ],
                        ),
                      ),
                      ValueListenableBuilder<bool>(
                        valueListenable: _modelLoaded,
                        builder: (context, loaded, _) => Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: loaded
                                ? scheme.secondaryContainer
                                : scheme.errorContainer,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(loaded ? Icons.check_circle : Icons.error,
                                  size: 16,
                                  color: loaded
                                      ? scheme.onSecondaryContainer
                                      : scheme.onErrorContainer),
                              const SizedBox(width: 6),
                              Text(loaded ? 'Model ready' : 'Model not loaded',
                                  style: textTheme.labelMedium?.copyWith(
                                    color: loaded
                                        ? scheme.onSecondaryContainer
                                        : scheme.onErrorContainer,
                                    fontWeight: FontWeight.w600,
                                  )),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),

                  ValueListenableBuilder<bool>(
                    valueListenable: _modelLoaded,
                    builder: (context, loaded, _) {
                      return ValueListenableBuilder<String?>(
                        valueListenable: _modelError,
                        builder: (context, modelErr, __) {
                          if (loaded || modelErr == null || modelErr.isEmpty) {
                            return const SizedBox.shrink();
                          }
                          return Padding(
                            padding: const EdgeInsets.only(top: 8),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 8),
                              decoration: BoxDecoration(
                                color: scheme.errorContainer.withAlpha(225),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                modelErr,
                                maxLines: 3,
                                overflow: TextOverflow.ellipsis,
                                style: textTheme.bodySmall?.copyWith(
                                  color: scheme.onErrorContainer,
                                ),
                              ),
                            ),
                          );
                        },
                      );
                    },
                  ),

                  const SizedBox(height: 16),

                  // Image preview card
                  Expanded(
                    child: Card(
                      clipBehavior: Clip.antiAlias,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              scheme.surfaceContainerHighest,
                              scheme.surface
                            ],
                          ),
                        ),
                        child: Center(
                          child: ValueListenableBuilder<Uint8List?>(
                            valueListenable: _imageBytes,
                            builder: (context, bytes, _) => AnimatedSwitcher(
                              duration: const Duration(milliseconds: 250),
                              child: bytes == null
                                  ? Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(Icons.image_outlined,
                                            size: 48, color: scheme.outline),
                                        const SizedBox(height: 12),
                                        Text('No image selected',
                                            style: textTheme.titleMedium
                                                ?.copyWith(
                                                    color: scheme.outline)),
                                        const SizedBox(height: 6),
                                        Text('Use Camera or Gallery below',
                                            style: textTheme.bodySmall
                                                ?.copyWith(
                                                    color: scheme.outline)),
                                      ],
                                    )
                                  : ClipRRect(
                                      borderRadius: BorderRadius.circular(0),
                                      child: Image.memory(bytes,
                                          fit: BoxFit.cover,
                                          width: double.infinity,
                                          height: double.infinity),
                                    ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Result card
                  ValueListenableBuilder<Map<String, dynamic>?>(
                    valueListenable: _lastDebug,
                    builder: (context, dbg, _) {
                      final hasErrors =
                          ((dbg?['errors'] as List?)?.isNotEmpty == true) ||
                              (dbg?['error'] != null);
                      return Card(
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                      hasErrors
                                          ? Icons.error_outline
                                          : Icons.eco_outlined,
                                      color: hasErrors
                                          ? scheme.error
                                          : scheme.primary),
                                  const SizedBox(width: 8),
                                  Text('Diagnosis',
                                      style: textTheme.titleMedium?.copyWith(
                                          fontWeight: FontWeight.w700)),
                                  const Spacer(),
                                  ValueListenableBuilder<bool>(
                                    valueListenable: _isLoading,
                                    builder: (context, loading, _) =>
                                        AnimatedSwitcher(
                                      duration:
                                          const Duration(milliseconds: 200),
                                      child: loading
                                          ? const SizedBox(
                                              height: 20,
                                              width: 20,
                                              child: CircularProgressIndicator(
                                                  strokeWidth: 2),
                                            )
                                          : const SizedBox.shrink(),
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: ValueListenableBuilder<String?>(
                                      valueListenable: _result,
                                      builder: (context, res, _) => Text(
                                        res ?? '—',
                                        style: textTheme.headlineSmall
                                            ?.copyWith(
                                                fontWeight: FontWeight.w700),
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  ValueListenableBuilder<double?>(
                                    valueListenable: _confidence,
                                    builder: (context, conf, _) {
                                      if (conf == null) {
                                        return const SizedBox.shrink();
                                      }
                                      final pct = (conf * 100)
                                          .clamp(0, 100)
                                          .toStringAsFixed(1);
                                      return Chip(
                                        label: Text('$pct%'),
                                        avatar: const Icon(
                                            Icons.insights_rounded,
                                            size: 18),
                                      );
                                    },
                                  ),
                                ],
                              ),
                              if (hasErrors) ...[
                                const SizedBox(height: 10),
                                Container(
                                  width: double.infinity,
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: scheme.errorContainer,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Text(
                                    ((dbg?['errors'] as List?)?.join('\n') ??
                                        (dbg?['error']?.toString() ??
                                            'Unknown inference error')),
                                    style: textTheme.bodySmall?.copyWith(
                                        color: scheme.onErrorContainer),
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 12),

                  // Action bar
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_rounded),
                          label: const Text('Gallery'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt_rounded),
                          label: const Text('Camera'),
                        ),
                      ),
                      // Test button removed
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
