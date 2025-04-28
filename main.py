import cv2
import time
import numpy as np
import handTracking as htm
import math
from ctypes import cast, POINTER, windll
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import psutil
import os
# Windows API for media control
import ctypes
from ctypes import wintypes

# Define Windows key codes for media control
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_VOLUME_MUTE = 0xAD

# Function to simulate media key presses with debounce protection
last_key_press = {
    VK_MEDIA_PLAY_PAUSE: 0,
    VK_MEDIA_NEXT_TRACK: 0,
    VK_MEDIA_PREV_TRACK: 0,
    VK_VOLUME_MUTE: 0
}
KEY_PRESS_DELAY = 1.5  # Increased delay to prevent double-skips

def press_media_key(key_code):
    global last_key_press
    current_time = time.time()
    
    # Add debounce protection to prevent unintended multiple presses
    if current_time - last_key_press.get(key_code, 0) < KEY_PRESS_DELAY:
        print(f"Ignoring key press (debounce): {key_code}")
        return False
    
    # Simulate key press
    windll.user32.keybd_event(key_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.1)  # Small delay between press and release for better recognition
    windll.user32.keybd_event(key_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
    
    # Update last press time
    last_key_press[key_code] = current_time
    return True

# Try to get current media info (basic implementation)
def get_current_song():
    try:
        # This is a simplified approach - in reality, we'd need to query specific media players
        # or use Windows API to get more accurate information
        import win32gui
        windows = []
        win32gui.EnumWindows(lambda hwnd, windows: windows.append((hwnd, win32gui.GetWindowText(hwnd))), windows)
        
        # Look for common media player windows
        media_players = {
            "spotify": "Spotify",
            "windows_media": "Windows Media Player",
            "vlc": "VLC",
            "chrome": "Chrome",
            "firefox": "Firefox",
            "edge": "Edge"
        }
        
        for hwnd, title in windows:
            for player, name in media_players.items():
                if name.lower() in title.lower() and " - " in title:
                    # Basic parsing of window title which often contains song info
                    parts = title.split(" - ")
                    if len(parts) >= 2:
                        # Format might be "Song - Artist - Player"
                        song_info = f"{parts[0]} - {parts[1]}"
                        return song_info[:30]  # Truncate if too long
        
        return "No song detected"
    except:
        return "Media info unavailable"

# Increase camera resolution
wCam, hCam = 1280, 720

# Better GPU acceleration setup with more options
print("Attempting to enable GPU acceleration...")
use_gpu = False
gpu_device = None
try:
    # Try CUDA first
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_devices > 0:
        print(f"CUDA enabled with {cuda_devices} device(s)")
        cv2.cuda.setDevice(0)  # Use first CUDA device
        gpu_device = cv2.cuda_GpuMat()
        use_gpu = True
    else:
        # Try OpenCL as fallback
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("OpenCL acceleration enabled")
                use_gpu = True
            else:
                print("OpenCL available but couldn't be enabled")
        else:
            print("No GPU acceleration available")
except Exception as e:
    print(f"GPU initialization error: {str(e)}")
    print("Falling back to CPU")

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Set optimal camera parameters
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for better performance

pTime = 0

# Initialize hand detector with optimized parameters
detector = htm.handDetector(detectionCon=0.7, maxHands=2)

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initial values
controlMode = "LOCKED"
isSystemMuted = volume.GetMute()
current_vol = int(volume.GetMasterVolumeLevelScalar() * 100)

# Music control variables
current_song_info = "No media detected"
media_action_feedback = ""
media_action_timeout = 0
song_update_timer = 0
song_update_interval = 5  # Update song info every 5 seconds
media_playing_state = False  # Track if media is playing

# Control modes
CONTROL_MODES = ["LOCKED", "VOLUME", "MEDIA"]
current_mode_index = 0  # Start in locked mode

# Smoothing variables for better stability
smoothingFactor = 5
lastVols = [current_vol] * smoothingFactor
lastHandPositions = []  # Store last hand positions for better volume stability
position_history_size = 5  # Number of positions to track

# Gesture stability variables
gestureCounters = {
    "lock": 0,
    "unlock": 0,
    "mode_switch": 0,
    "play": 0,
    "stop": 0,
    "next_track": 0,
    "prev_track": 0
}
requiredFrames = 10
lastGestureTime = time.time()
gestureStabilityDelay = 0.8

# Performance monitoring
fps_values = []
cpu_values = []

# Function to apply minimal overlay without excessive drawing
def draw_minimal_overlay(img, mode_text, mode_color):
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Semi-transparent overlay for status
    status_overlay = img[h//2-30:h//2+30, w//2-150:w//2+150].copy()
    cv2.rectangle(img, (w//2-150, h//2-30), (w//2+150, h//2+30), (0, 0, 0), -1)
    cv2.addWeighted(status_overlay, 0.2, img[h//2-30:h//2+30, w//2-150:w//2+150], 0.8, 0, img[h//2-30:h//2+30, w//2-150:w//2+150])
    
    # Mode text in center
    cv2.putText(img, mode_text, (w//2-110, h//2+10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
    
    # Status bar at the bottom for controls and song info
    cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
    
    # Draw song info if in media mode
    if "MEDIA" in mode_text:
        cv2.putText(img, f"Now Playing: {current_song_info}", 
                   (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    
    # Show media action feedback
    if time.time() < media_action_timeout:
        cv2.putText(img, media_action_feedback, 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Return the modified image
    return img

# Function to draw a simple volume bar
def draw_simple_volume_bar(img, percentage):
    h, w = img.shape[:2]
    bar_width = 30
    bar_height = 200
    start_x = w - bar_width - 20
    start_y = (h - bar_height) // 2
    
    # Draw background bar
    cv2.rectangle(img, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (50, 50, 50), -1)
    
    # Draw filled portion
    filled_height = int(bar_height * percentage / 100)
    if filled_height > 0:
        # Color based on volume level
        if percentage < 30:
            color = (0, 150, 255)  # Orange for low
        elif percentage < 70:
            color = (0, 255, 255)  # Yellow for medium
        else:
            color = (0, 255, 0)    # Green for high
            
        cv2.rectangle(img, 
                     (start_x, start_y + bar_height - filled_height), 
                     (start_x + bar_width, start_y + bar_height), 
                     color, -1)
    
    # Draw volume percentage
    cv2.putText(img, f"{int(percentage)}%", 
                (start_x - 10, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

# Optimized gesture detection function with expanded gestures for media control
def detect_gestures(fingers_left, fingers_right, lmList_left=None, lmList_right=None):
    gestures = {
        "lock": False,
        "unlock": False,
        "mode_switch": False,
        "volume_control": False,
        "play": False,
        "stop": False,
        "next_track": False,
        "prev_track": False
    }
    
    # LEFT HAND GESTURES - CONTROL FUNCTIONS
    if fingers_left:
        # Peace sign (index and middle fingers up) for lock
        if fingers_left[1] == 1 and fingers_left[2] == 1 and fingers_left[0] == 0 and fingers_left[3] == 0 and fingers_left[4] == 0:
            gestures["lock"] = True
        
        # Open palm (all fingers up) for unlock
        if fingers_left.count(1) >= 4:  # More tolerant check
            gestures["unlock"] = True
            
        # Mode switch - Thumb and pinky up (like a "phone" gesture)
        if fingers_left[0] == 1 and fingers_left[4] == 1 and fingers_left[1] == 0 and fingers_left[2] == 0 and fingers_left[3] == 0:
            gestures["mode_switch"] = True
    
    # RIGHT HAND GESTURES - FUNCTIONAL CONTROLS
    if fingers_right:
        # Volume control - Pinch gesture (thumb and index)
        if fingers_right[0] == 1 and fingers_right[1] == 1:
            gestures["volume_control"] = True
            
        # Media controls - only active in MEDIA mode
        # PLAY - Thumbs up (only thumb extended)
        if fingers_right[0] == 1 and fingers_right[1] == 0 and fingers_right[2] == 0 and fingers_right[3] == 0 and fingers_right[4] == 0:
            gestures["play"] = True
        
        # STOP - Fist (no fingers extended)
        if fingers_right[0] == 0 and fingers_right[1] == 0 and fingers_right[2] == 0 and fingers_right[3] == 0 and fingers_right[4] == 0:
            gestures["stop"] = True
            
        # Next track - Swipe right (index and middle fingers extended)
        if fingers_right[1] == 1 and fingers_right[2] == 1 and fingers_right[0] == 0 and fingers_right[3] == 0 and fingers_right[4] == 0:
            gestures["next_track"] = True
            
        # Previous track - Index finger only
        if fingers_right[1] == 1 and fingers_right[0] == 0 and fingers_right[2] == 0 and fingers_right[3] == 0 and fingers_right[4] == 0:
            gestures["prev_track"] = True
    
    return gestures

# Function to calculate volume based on hand position with improved accuracy
def calculate_volume_from_hand(thumb_tip, index_tip):
    # Calculate distance between thumb and index finger
    distance = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
    
    # Get absolute hand position (vertical position in frame)
    hand_height = min(thumb_tip[2], index_tip[2])  # Y-position (lower value = higher in frame)
    
    # Use both distance AND position for volume calculation
    h, w = img.shape[:2]  # Get frame dimensions
    
    # Map vertical position to a base volume (higher in frame = higher volume)
    base_vol = np.interp(hand_height, [h//4, h*3//4], [80, 20])
    
    # Map pinch distance to a volume adjustment
    vol_adjustment = np.interp(distance, [20, 150], [-20, 20])
    
    # Combine for final volume
    vol = base_vol + vol_adjustment
    vol = np.clip(vol, 0, 100)  # Ensure volume stays within range
    
    return vol, distance

# Main loop
showHelp = True
helpTimeout = time.time() + 5  # Only show help briefly
last_frame_time = time.time()

while True:
    # Track CPU usage
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_values.append(cpu_percent)
    if len(cpu_values) > 5:  # Keep last 5 values
        cpu_values.pop(0)
    
    # Calculate time for FPS limiting if needed
    current_time = time.time()
    elapsed = current_time - last_frame_time
    
    # Try to maintain consistent frame processing rate
    if elapsed < 1/30:  # Cap at 30 FPS to save CPU
        time.sleep(1/30 - elapsed)
    
    last_frame_time = time.time()
    
    # Update song info periodically
    if current_time - song_update_timer > song_update_interval:
        current_song_info = get_current_song()
        song_update_timer = current_time
    
    # Read frame
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break
    
    # Process with GPU if available
    if use_gpu and hasattr(cv2, 'cuda') and 'cuda_devices' in locals() and cuda_devices > 0:
        try:
            # Upload to GPU, process, download (only for operations that support CUDA)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(img)
            
            # Flip on GPU if possible
            img_flipped = cv2.cuda.flip(gpu_frame, 1)
            img = img_flipped.download()
        except Exception as e:
            print(f"GPU processing error: {e}")
            # Fall back to CPU
            img = cv2.flip(img, 1)
    else:
        # Process on CPU
        img = cv2.flip(img, 1)
    
    # Find Hands - optimized to reduce processing
    img = detector.findHands(img, draw=True)  # Keep hand landmarks visible
    
    # Process hands separately
    left_hand = {"detected": False, "fingers": [], "lmList": [], "bbox": None}
    right_hand = {"detected": False, "fingers": [], "lmList": [], "bbox": None}
    
    if hasattr(detector, 'results') and detector.results.multi_hand_landmarks:
        multi_hand_landmarks = detector.results.multi_hand_landmarks
        multi_handedness = detector.results.multi_handedness
        
        # Process each detected hand
        for idx, hand in enumerate(multi_handedness):
            if idx >= 2:  # Process max 2 hands
                break
                
            # Determine hand laterality (after mirroring)
            is_left_in_image = hand.classification[0].label == "Right"  # Flipped due to mirror
            
            # Extract landmarks
            lmList = []
            xList, yList = [], []
            
            hand_landmarks = multi_hand_landmarks[idx]
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)
            
            # Calculate bounding box
            if xList and yList:
                bbox = (min(xList), min(yList), max(xList), max(yList))
                
                # Calculate fingers
                fingers = []
                # Thumb
                if lmList[4][1] > lmList[4-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # Fingers
                for finger_id in range(1, 5):
                    tip_id = detector.tipIds[finger_id]
                    if lmList[tip_id][2] < lmList[tip_id-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # Store data based on which hand it is
                if is_left_in_image:
                    left_hand = {
                        "detected": True, 
                        "fingers": fingers,
                        "lmList": lmList,
                        "bbox": bbox
                    }
                    # Label left hand with minimal UI
                    cv2.putText(img, "L", (bbox[0], bbox[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    right_hand = {
                        "detected": True,
                        "fingers": fingers,
                        "lmList": lmList,
                        "bbox": bbox
                    }
                    # Label right hand with minimal UI
                    cv2.putText(img, "R", (bbox[0], bbox[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Detect gestures based on both hands
    gestures = detect_gestures(
        left_hand["fingers"] if left_hand["detected"] else [],
        right_hand["fingers"] if right_hand["detected"] else []
    )
    
    # Update gesture counters
    for gesture_name, is_active in gestures.items():
        if is_active:
            gestureCounters[gesture_name] = min(gestureCounters.get(gesture_name, 0) + 1, requiredFrames + 5)
            if left_hand["detected"] and gesture_name in ["lock", "unlock", "mode_switch"]:
                cv2.putText(img, f"{gesture_name.replace('_', ' ').title()}...", 
                            (left_hand["bbox"][0], left_hand["bbox"][1]-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 0, 255) if gesture_name == "lock" else (0, 255, 0), 2)
        else:
            gestureCounters[gesture_name] = 0
    
    # Process control gestures
    if time.time() - lastGestureTime > gestureStabilityDelay:
        # Process lock/unlock
        if gestureCounters["lock"] > requiredFrames:
            controlMode = "LOCKED"
            current_mode_index = 0
            lastGestureTime = time.time()
            gestureCounters["lock"] = 0
            media_action_feedback = "System Locked"
            media_action_timeout = time.time() + 2
            
        elif gestureCounters["unlock"] > requiredFrames and controlMode == "LOCKED":
            controlMode = "VOLUME"  # Default to volume mode when unlocked
            current_mode_index = 1
            lastGestureTime = time.time()
            gestureCounters["unlock"] = 0
            media_action_feedback = "System Unlocked - Volume Mode"
            media_action_timeout = time.time() + 2
            
        # Mode switching (only when unlocked)
        elif gestureCounters["mode_switch"] > requiredFrames and controlMode != "LOCKED":
            # Cycle through modes: VOLUME -> MEDIA -> VOLUME -> ...
            current_mode_index = (current_mode_index + 1) % len(CONTROL_MODES)
            if current_mode_index == 0:  # Skip LOCKED mode during cycling
                current_mode_index = 1
            controlMode = CONTROL_MODES[current_mode_index]
            lastGestureTime = time.time()
            gestureCounters["mode_switch"] = 0
            media_action_feedback = f"Switched to {controlMode} Mode"
            media_action_timeout = time.time() + 2
        
        # Process media controls (only in MEDIA mode)
        elif controlMode == "MEDIA":
            # Play - Thumbs up
            if gestureCounters["play"] > requiredFrames:
                if press_media_key(VK_MEDIA_PLAY_PAUSE):
                    media_playing_state = True
                    lastGestureTime = time.time()
                    gestureCounters["play"] = 0
                    media_action_feedback = "Play"
                    media_action_timeout = time.time() + 1.5
            
            # Stop - Fist gesture
            elif gestureCounters["stop"] > requiredFrames:
                if press_media_key(VK_MEDIA_PLAY_PAUSE):
                    media_playing_state = False
                    lastGestureTime = time.time()
                    gestureCounters["stop"] = 0
                    media_action_feedback = "Stop/Pause"
                    media_action_timeout = time.time() + 1.5
                
            # Next Track - with enhanced debounce protection
            elif gestureCounters["next_track"] > requiredFrames:
                if press_media_key(VK_MEDIA_NEXT_TRACK):
                    lastGestureTime = time.time() + 0.5  # Add extra delay for skip operations
                    gestureCounters["next_track"] = 0
                    media_action_feedback = "Next Track"
                    media_action_timeout = time.time() + 1.5
                    # Force song info update
                    current_song_info = "Updating..."
                    song_update_timer = 0
                
            # Previous Track - with enhanced debounce protection
            elif gestureCounters["prev_track"] > requiredFrames:
                if press_media_key(VK_MEDIA_PREV_TRACK):
                    lastGestureTime = time.time() + 0.5  # Add extra delay for skip operations
                    gestureCounters["prev_track"] = 0
                    media_action_feedback = "Previous Track"
                    media_action_timeout = time.time() + 1.5
                    # Force song info update
                    current_song_info = "Updating..."
                    song_update_timer = 0
    
    # Process volume control with right hand if in VOLUME mode
    if controlMode == "VOLUME" and gestures["volume_control"] and right_hand["detected"]:
        # Get pinch points
        thumb_tip = right_hand["lmList"][4]
        index_tip = right_hand["lmList"][8]
        
        # Calculate volume with improved accuracy
        volPer, distance = calculate_volume_from_hand(thumb_tip, index_tip)
        
        # Track hand position for stability
        current_position = [(thumb_tip[1] + index_tip[1])//2, (thumb_tip[2] + index_tip[2])//2]
        lastHandPositions.append(current_position)
        if len(lastHandPositions) > position_history_size:
            lastHandPositions.pop(0)
        
        # Initialize variance values
        x_variance = 0
        y_variance = 0
        
        # Only change volume if hand position is relatively stable
        if len(lastHandPositions) >= 3:
            # Calculate variance in hand position
            x_positions = [pos[0] for pos in lastHandPositions]
            y_positions = [pos[1] for pos in lastHandPositions]
            x_variance = max(x_positions) - min(x_positions)
            y_variance = max(y_positions) - min(y_positions)
            
            # If hand is stable (low variance), apply volume change
            if x_variance < 30 and y_variance < 30:
                # Apply smoothing
                lastVols.pop(0)
                lastVols.append(volPer)
                smoothed_vol = sum(lastVols) / len(lastVols)
                
                # Only update volume if there's a significant change
                current_vol_pct = int(volume.GetMasterVolumeLevelScalar() * 100)
                if abs(smoothed_vol - current_vol_pct) > 1:  # Threshold for volume change
                    # Set system volume
                    volume.SetMasterVolumeLevelScalar(smoothed_vol / 100, None)
        
        # Visual feedback for volume control with color gradient based on stability
        stability = min(1.0, 50.0 / (1 + x_variance + y_variance))
        line_color = (
            int(255 * (1-stability)),  # B
            int(255 * stability),      # G
            0                          # R
        )
        cv2.line(img, (thumb_tip[1], thumb_tip[2]), (index_tip[1], index_tip[2]), line_color, 3)
        cx, cy = (thumb_tip[1] + index_tip[1]) // 2, (thumb_tip[2] + index_tip[2]) // 2
        cv2.circle(img, (cx, cy), 10, line_color, cv2.FILLED)
        
        # Show calculated volume for feedback
        cv2.putText(img, f"Vol: {int(volPer)}%", (cx - 40, cy - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Get current system volume
    current_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    isSystemMuted = volume.GetMute()
    
    # Draw minimal overlay with status based on control mode
    mode_color = (0, 0, 255) if controlMode == "LOCKED" else (0, 255, 255) if controlMode == "MEDIA" else (0, 255, 0)
    img = draw_minimal_overlay(img, controlMode, mode_color)
    
    # Draw controls based on mode
    if controlMode == "VOLUME":
        if isSystemMuted:
            cv2.putText(img, "MUTED", (img.shape[1]-100, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            img = draw_simple_volume_bar(img, current_vol)
    elif controlMode == "MEDIA":
        # Draw media control indicators
        h, w = img.shape[:2]
        # Top right area - show media gesture guides
        cv2.rectangle(img, (w-230, 10), (w-10, 110), (40, 40, 40), -1)
        cv2.putText(img, "Media Controls:", (w-220, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Thumbs Up: Play", (w-220, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Fist: Pause/Stop", (w-220, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Peace: Next Track", (w-220, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Index: Previous Track", (w-220, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show brief help overlay when starting
    if showHelp and time.time() < helpTimeout:
        h, w = img.shape[:2]
        cv2.rectangle(img, (w//2-250, 100), (w//2+250, 210), (0, 0, 0), -1)
        cv2.putText(img, "LEFT HAND CONTROLS:", (w//2-240, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- Peace sign: Lock system", (w//2-240, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "- Open hand: Unlock system", (w//2-240, 165), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "- Thumb+Pinky: Switch modes (Volume/Media)", (w//2-240, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Press 'q' to exit", (w//2-240, 205), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    fps_values.append(fps)
    if len(fps_values) > 5:
        fps_values.pop(0)
    avg_fps = sum(fps_values) / len(fps_values)
    avg_cpu = sum(cpu_values) / len(cpu_values)
    
    # Draw performance stats in bottom left
    h, w = img.shape[:2]
    cv2.putText(img, f'FPS: {int(avg_fps)} | GPU: {"ON" if use_gpu else "OFF"}', 
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if use_gpu else (0, 255, 255), 2)
    
    # Show image
    cv2.imshow("Hand Gesture Media Control", img)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()