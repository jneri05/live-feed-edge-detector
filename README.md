# Live feed line detector

## Introduction
Applying Canny to live video feed to detect edges for each frame. Implements Hough Transform and Total Least Squares to show different techniques of line fitting on detected edges. The output is a window with 4 panels showing original feed and each different technique output applied on each frame of the live video.

## Output Example
Screenshot of a frame detecting edges of a racket

<img width="445" height="340" alt="image" src="https://github.com/user-attachments/assets/f82921db-2291-41e3-aa2e-5b3c24ff9b78" />

## How to run on VS Code
1. install requirements with `"pip install opencv-python numpy"`
2. In `main.py` check line `cap = cv2.VideoCapture(0)`, 0 will use your default webcam. Change to 1 if you're using an external camera
3. Run `main.py`
4. A window appears showing live camera feed and detection results for each frame
5. Press `q` to quit

## Known Issues
- If camera feed environment is blurry, cluttered, or has low light then edge detection will miss lines or create false lines

## Requirements
- Python 3
- `opencv-python`
- `numpy`
