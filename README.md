# Gesture-Based Media Control System

A computer vision application that allows users to control media playback and system volume using hand gestures, built with Python, OpenCV, and MediaPipe.

![Demo - Add Screenshot/GIF here](path/to/demo.gif)

## Features

- **Real-Time Hand Tracking**: Detects and tracks hand movements with MediaPipe
- **Multiple Control Modes**:
  - **Volume Mode**: Control system volume with hand gestures
  - **Media Mode**: Control media playback with gestures
  - **Locked Mode**: Prevent accidental input
- **Intuitive Gesture Controls**:
  - Adjust volume by pinching gestures
  - Play/pause with thumbs up/fist gestures
  - Skip tracks with simple finger positions
  - Switch modes with specific hand positions
- **Visual Feedback**: On-screen display of current mode, volume levels, and media information
- **GPU Acceleration**: Utilizes GPU processing when available for better performance

## Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- PyCaw (for Windows audio control)
- PySound (for audio processing)
- Comtypes (for Windows API integration)
- psutil (for performance monitoring)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nour23019870/gesture-media-control.git
   cd gesture-media-control
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On Unix/MacOS:
   source env/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Position yourself in front of your webcam.

3. Use the following gestures:

   ### Left Hand Gestures (Control System)
   - **Peace Sign (✌️)**: Lock the system
   - **Open Palm (🖐️)**: Unlock the system
   - **Thumb + Pinky ("Phone" gesture 🤙)**: Switch between Volume and Media modes

   ### Right Hand Gestures
   - **Volume Mode**:
     - Pinch gesture (👌) and move up/down to adjust volume

   - **Media Mode**:
     - **Thumbs Up (👍)**: Play
     - **Fist (✊)**: Pause/Stop
     - **Peace Sign (✌️)**: Next Track
     - **Index Finger (☝️)**: Previous Track

4. Press 'q' to exit the application.

## Project Structure

- `main.py`: Main application logic and UI
- `handTracking.py`: Hand detection and tracking module
- `volumeControl.py`: Volume control functionality module

## Performance Tips

- The application will attempt to use GPU acceleration if available
- Close other applications for optimal performance
- Ensure proper lighting for better hand detection

## License

[Add your license information here]

## Acknowledgments

- MediaPipe team for their hand tracking implementation
- OpenCV for computer vision capabilities
- PyCaw for Windows audio integration

---

*Note: This project currently supports Windows operating systems.*