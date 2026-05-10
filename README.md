# Smart Data Extraction

This folder contains a Playwright + YOLO pipeline that:
- Captures clean, scrolled screenshots from a web page.
- Detects document layout regions (text, tables, figures, headers, etc.).
- Saves annotated screenshots and cropped elements for each label.

## Folder layout
- smart_screenshots/ - Raw screenshots captured from the page.
- output2/ - Per-screenshot outputs and annotated images.
- models/ - Local model file location (ignored by git).

## Setup
Install dependencies:

```
pip install playwright ultralytics pillow numpy
python -m playwright install
```

## Model
Place the YOLO DocLayNet model here:

```
models/yolov11l-doclaynet.pt
```

Then update the model path in yolo3.py to point to this file.

## Run
```
python yolo3.py
```

Outputs will be written to smart_screenshots/ and output2/.

## Notes
The script has a default URL set near the bottom of yolo3.py. Change it to process a different page.
