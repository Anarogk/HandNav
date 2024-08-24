# Hand Gesture Controller

A Python-based hand gesture recognition system that uses a webcam to capture hand movements and translates them into various system control commands, such as mouse movements, clicks, scrolls, volume control, and brightness adjustment. The system leverages OpenCV, MediaPipe, and PyAutoGUI libraries for computer vision and system control functionalities.

## Features

- **Hand Gesture Recognition**: Detects various hand gestures using MediaPipe and classifies them into different actions.
- **Mouse Control**: Move the cursor, click, double-click, and right-click using specific hand gestures.
- **Scroll Control**: Scroll vertically and horizontally by pinching gestures.
- **Volume Control**: Adjust the system volume using pinch gestures.
- **Brightness Control**: Adjust the system brightness using pinch gestures.
- **Dual Hand Recognition**: Supports recognizing and distinguishing between gestures made with the left and right hands.

## Dependencies

- Python 3.6+
- OpenCV
- MediaPipe
- PyAutoGUI
- Screen Brightness Control

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/hand-gesture-controller.git
cd hand-gesture-controller
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Gesture Controller:**

```bash
python gesture_controller.py
```

## Project Structure

- `gesture_controller.py`: Main entry point of the program. Captures video frames, processes them, and handles gestures.
- `hand_recog.py`: Contains the `HandRecog` class that converts MediaPipe landmarks to recognizable gestures.
- `controller.py`: Contains the `Controller` class that executes commands according to detected gestures.
- `requirements.txt`: List of dependencies.

## Usage

- **Run the program**: After starting the program, it will open a webcam window.
- **Perform gestures**: Use the following gestures to control the system:
  - **Move Cursor**: Show a "V" gesture and move your hand to control the mouse cursor.
  - **Left Click**: Make a fist to grab and release to click.
  - **Right Click**: Point with the index finger.
  - **Double Click**: Close two fingers (index and middle).
  - **Scroll**: Pinch with the minor hand to scroll.
  - **Adjust Volume/Brightness**: Pinch with the major hand to control volume or brightness based on the direction of the pinch.

## Hand Gestures

- **Fist**: Left-click and drag.
- **Index Finger Pointing**: Right-click.
- **Two Fingers Pointing (V Gesture)**: Move the cursor.
- **Two Fingers Closed**: Double-click.
- **Pinch (Major Hand)**: Adjust brightness/volume.
- **Pinch (Minor Hand)**: Scroll.

## How It Works

1. **Capture Frames**: The webcam captures frames continuously.
2. **Hand Detection**: MediaPipe detects hand landmarks in the captured frames.
3. **Gesture Recognition**: The `HandRecog` class processes these landmarks to recognize specific gestures.
4. **Command Execution**: Based on the recognized gestures, the `Controller` class executes system commands like moving the cursor, clicking, scrolling, or adjusting volume/brightness.

## Future Improvements

- Improve gesture recognition accuracy.
- Add support for more gestures and system controls.
- Enhance performance for real-time applications.
- Implement custom gesture training for personalized commands.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google for hand landmark detection.
- [OpenCV](https://opencv.org/) for image processing.
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for controlling the mouse and keyboard.
- [Screen Brightness Control](https://github.com/Crozzers/screen-brightness-control) for adjusting screen brightness.
