# Hand Gesture Recognition — web demo

A camera-only browser demo that recognises the same six gestures as the
Python app:

- Open palm
- Fist
- Thumbs up
- Peace
- Pointing
- Three fingers

MediaPipe finds the 21 hand landmarks in the browser. The trained Random
Forest from `models/gesture_model.pkl` is exported to JSON and evaluated in
JavaScript, so no video frames or predictions leave the machine.

Like the desktop app, a sound plays when a new gesture is recognised. The
**Sound** row in the readout can mute playback or change the volume.

## Run it

```bash
# 1. Export the trained model for the browser (one-time, or after retraining)
uv run export_model.py

# 2. Serve the repository root
uv run python -m http.server 8000
```

Open <http://localhost:8000/web/> and click **Enable camera**.

Serve the repository root as shown above so the WAV files in `sounds/` are
available to the page. Camera access requires `localhost` or an HTTPS page.
Chrome or Edge is recommended. The first load downloads MediaPipe's WASM
runtime and the hand landmarker model from their public CDNs, so it needs an
internet connection.

## Verify the JS model

```bash
node web/test_model.mjs
```

The script runs several real dataset rows through the exported forest and
checks that the class and probabilities match scikit-learn.

## How detection works

1. Each camera frame is drawn mirrored to a canvas, matching the frame flip in
   `gesture_recognition.py`.
2. MediaPipe's `HandLandmarker` returns 21 `(x, y, z)` landmarks.
3. Features are normalised exactly like `collect_data.py`: subtract the wrist,
   then divide by the largest xy distance from the wrist.
4. Every one of the 200 trees votes with its leaf class distribution; the
   averaged result is the prediction.
5. Predictions below 65% confidence become `UNKNOWN`, and the last 10 frames
   are smoothed by majority vote.
6. When the smoothed gesture changes, the matching WAV from `../sounds/` plays
   once. Repeated gestures stay silent and a 1-second cooldown matches
   `gesture_recognition.py`. The Sound control mutes audio or adjusts volume.

If retraining changes the number of gestures or features, re-run
`uv run export_model.py` and update `GESTURE_LABELS` and `SOUND_FILES` in
`web/app.js`.
