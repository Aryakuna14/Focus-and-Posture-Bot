# Security and Privacy

Project Angelina uses a webcam, local ML artifacts, a local web dashboard, and optional serial output to an ESP32. Treat those surfaces with care.

## Data Handling

- Webcam frames are processed locally by OpenCV and MediaPipe.
- The inference scripts send only posture labels, confidence scores, and alert state to the local dashboard endpoint.
- The dashboard binds to `127.0.0.1` by default. Set `ANGELINA_DASHBOARD_HOST=0.0.0.0` only if you intentionally want LAN access.
- Generated datasets and model artifacts are ignored by Git so personal posture data does not get committed by accident.

## Model Artifact Safety

`joblib` and pickle-based files can execute code while loading. For that reason, this repository does not ship generated `.pkl`, `.keras`, or `.h5` model artifacts.

Recommended workflow:

1. Create your own dataset with `1_data_collector.py`.
2. Train local artifacts with `2_train_svm.py` or `2_train_cnn.py`.
3. Run inference only with artifacts you created or explicitly trust.

## Hardware Safety

The ESP32 sketch can activate a buzzer and vibration motor. Use current-limiting components, test at low duty cycles, and disconnect hardware immediately if it heats, behaves unexpectedly, or causes discomfort.

## Reporting Issues

If you find a security issue, open a GitHub issue with reproduction details and avoid posting private webcam data, model artifacts, or personal datasets.
