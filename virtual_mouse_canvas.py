import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False 
import numpy as np
import time
import speech_recognition as sr
import webbrowser
import threading
from collections import deque

# Constants
PINCH_THRESHOLD = 0.05
SCROLL_ACTIVATION_THRESHOLD = 0.1
SCROLL_DEADZONE = 5
SCROLL_SMOOTHING = 5
SCROLL_SPEED = 0.3
VOICE_COMMAND = "google"  # Customize your wake word
FIST_THRESHOLD = 0.15  # Threshold for fist detection
DRAWING_THRESHOLD = 0.02  # Minimum movement to continue drawing
ERASER_SIZE = 30  # Size of eraser

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()
print(f"Screen dimensions: {screen_w} x {screen_h}")

# Canvas setup
canvas = None
canvas_width, canvas_height = 800, 600
drawing_color = (255, 0, 0)  # Blue color for drawing
drawing_thickness = 5
eraser_thickness = 30

# Voice recognition setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_for_commands():
    """Background thread to listen for voice commands"""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=2)
                command = recognizer.recognize_google(audio).lower()
                if VOICE_COMMAND in command:
                    webbrowser.open("https://www.google.com")
                    print("Voice command activated: Launched Google")
            except (sr.UnknownValueError, sr.RequestError):
                pass
            except Exception as e:
                print(f"Voice recognition error: {e}")

# Start voice thread
voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
voice_thread.start()

# Camera setup
cap = cv2.VideoCapture(0)

# State variables
prev_x, prev_y = 0, 0
left_click = False
right_click = False
scroll_active = False
scroll_buffer = []
scroll_reference_y = 0

# Canvas state variables
canvas_mode = False  # False = mouse mode, True = canvas mode
drawing_active = False
erasing_active = False
last_draw_x, last_draw_y = None, None
prev_hand_landmarks = None  # To detect hand stop

def is_fist(hand_landmarks):
    """Detect if hand is in fist position"""
    # Get all finger tips and their corresponding PIP joints
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    pip_ids = [3, 6, 10, 14, 18]   # Corresponding PIP joints
    
    fingers_curled = 0
    
    for tip_id, pip_id in zip(tips_ids[1:], pip_ids[1:]):  # Skip thumb for now
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        if tip.y > pip.y:  # Finger is curled (tip below PIP)
            fingers_curled += 1
    
    # Special check for thumb (curled if closer to palm)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_mcp = hand_landmarks.landmark[5]
    
    thumb_curled = abs(thumb_tip.x - index_mcp.x) < FIST_THRESHOLD and thumb_tip.y > thumb_ip.y
    
    return (fingers_curled >= 4) or (fingers_curled >= 3 and thumb_curled)

def is_drawing_gesture(hand_landmarks):
    """Check if only index finger is extended (drawing mode)"""
    # Index finger extended
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_extended = index_tip.y < index_pip.y
    
    # All other fingers curled
    other_fingers_curled = True
    for tip_id, pip_id in [(12,10), (16,14), (20,18)]:  # Middle, Ring, Pinky
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        if tip.y < pip.y:  # Finger extended
            other_fingers_curled = False
            break
    
    return index_extended and other_fingers_curled

def is_erasing_gesture(hand_landmarks):
    """Check if index and middle fingers are extended (erasing mode)"""
    # Index and middle extended
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]
    
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    
    # Ring and pinky curled
    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]
    
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y
    
    return index_extended and middle_extended and ring_curled and pinky_curled

def is_clear_gesture(hand_landmarks):
    """Check if thumb and index finger are pinched (clear canvas)"""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    distance = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return distance < PINCH_THRESHOLD

def is_scroll_gesture(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    
    if (ring_tip.y < hand_landmarks.landmark[13].y or 
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[17].y):
        return False
    return np.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y) < SCROLL_ACTIVATION_THRESHOLD

# Initialize canvas
def init_canvas():
    global canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    cv2.namedWindow("Virtual Canvas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Canvas", canvas_width, canvas_height)

def clear_canvas():
    global canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Initialize canvas
init_canvas()

# Fist toggle debouncing
last_fist_time = 0
fist_cooldown = 1.0  # 1 second cooldown for fist toggles

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Get hand landmarks if available
    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Check for fist toggle (with debouncing)
        current_time = time.time()
        if is_fist(hand_landmarks) and (current_time - last_fist_time) > fist_cooldown:
            canvas_mode = not canvas_mode
            last_fist_time = current_time
            print(f"Switched to {'Canvas' if canvas_mode else 'Mouse'} mode")
            # Reset states when switching modes
            drawing_active = False
            erasing_active = False
            last_draw_x, last_draw_y = None, None
        
        if canvas_mode:
            # Canvas mode logic
            if hand_landmarks:
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                
                # Map to canvas coordinates
                canvas_x = int(index_tip.x * canvas_width)
                canvas_y = int(index_tip.y * canvas_height)
                
                # Constrain to canvas boundaries
                canvas_x = max(0, min(canvas_x, canvas_width - 1))
                canvas_y = max(0, min(canvas_y, canvas_height - 1))
                
                # Check hand movement (stop drawing if hand stops)
                hand_moving = True
                if prev_hand_landmarks:
                    movement = np.hypot(
                        index_tip.x - prev_hand_landmarks.landmark[8].x,
                        index_tip.y - prev_hand_landmarks.landmark[8].y
                    )
                    hand_moving = movement > DRAWING_THRESHOLD
                
                # Check gestures
                if is_clear_gesture(hand_landmarks):
                    clear_canvas()
                    drawing_active = False
                    erasing_active = False
                elif is_erasing_gesture(hand_landmarks):
                    erasing_active = True
                    drawing_active = False
                    if hand_moving and last_draw_x is not None and last_draw_y is not None:
                        # Draw eraser line
                        cv2.line(canvas, 
                                (last_draw_x, last_draw_y),
                                (canvas_x, canvas_y), 
                                (255, 255, 255),  # White color (erase)
                                eraser_thickness)
                    last_draw_x, last_draw_y = canvas_x, canvas_y
                elif is_drawing_gesture(hand_landmarks):
                    drawing_active = True
                    erasing_active = False
                    if hand_moving and last_draw_x is not None and last_draw_y is not None:
                        # Draw line
                        cv2.line(canvas, 
                                (last_draw_x, last_draw_y), 
                                (canvas_x, canvas_y), 
                                drawing_color, 
                                drawing_thickness)
                    last_draw_x, last_draw_y = canvas_x, canvas_y
                else:
                    # No drawing/erasing gesture detected
                    drawing_active = False
                    erasing_active = False
                    last_draw_x, last_draw_y = None, None
                
                # Draw cursor on canvas based on mode
                cursor_color = (0, 255, 0) if drawing_active else (0, 0, 255) if erasing_active else (128, 128, 128)
                cv2.circle(canvas, (canvas_x, canvas_y), 5, cursor_color, -1)
                
                # Store current landmarks for next frame
                prev_hand_landmarks = hand_landmarks
                
                # Draw landmarks on camera feed
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        else:
            # Mouse mode logic (your existing code)
            # Get landmarks
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]
            
            # Cursor movement
            curr_x = int(index_tip.x * screen_w)
            curr_y = int(index_tip.y * screen_h)
            smoothed_x = prev_x + (curr_x - prev_x) * 0.2
            smoothed_y = prev_y + (curr_y - prev_y) * 0.2
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y

            # Scroll gesture
            if is_scroll_gesture(hand_landmarks):
                if not scroll_active:
                    scroll_active = True
                    scroll_reference_y = index_tip.y
                    scroll_buffer = []
                
                delta_y = (index_tip.y - scroll_reference_y) * screen_h
                scroll_buffer.append(delta_y)
                if len(scroll_buffer) > SCROLL_SMOOTHING:
                    scroll_buffer.pop(0)
                
                if scroll_buffer and abs(delta_y) > SCROLL_DEADZONE:
                    pyautogui.scroll(int(-delta_y * SCROLL_SPEED))
                    scroll_reference_y = index_tip.y
            else:
                scroll_active = False
                
                # Click detection
                left_pinch = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                right_pinch = np.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
                
                if left_pinch < PINCH_THRESHOLD:
                    if not left_click:
                        pyautogui.mouseDown(button='left')
                        left_click = True
                elif left_click:
                    pyautogui.mouseUp(button='left')
                    left_click = False
                
                if right_pinch < PINCH_THRESHOLD:
                    if not right_click:
                        pyautogui.mouseDown(button='right')
                        right_click = True
                elif right_click:
                    pyautogui.mouseUp(button='right')
                    right_click = False

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # No hand detected
        prev_hand_landmarks = None
        if canvas_mode:
            drawing_active = False
            erasing_active = False
            last_draw_x, last_draw_y = None, None

    # Display status on camera feed
    status_text = []
    mode_text = "CANVAS MODE" if canvas_mode else "MOUSE MODE"
    status_text.append(mode_text)
    
    if canvas_mode:
        if drawing_active:
            status_text.append("DRAWING")
        elif erasing_active:
            status_text.append("ERASING")
        else:
            status_text.append("READY")
    else:
        if left_click:
            status_text.append("LEFT CLICK")
        if right_click:
            status_text.append("RIGHT CLICK")
        if scroll_active:
            status_text.append("SCROLLING")
    
    for i, text in enumerate(status_text):
        color = (0,255,0) if "CANVAS" in text else (255,0,0) if "MOUSE" in text else \
                (255,0,0) if "DRAWING" in text else (0,0,255) if "ERASING" in text else \
                (0,255,255) if "CLICK" in text else (255,255,0) if "SCROLLING" in text else (255,255,255)
        cv2.putText(frame, text, (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Instructions
    instructions = [
        "Fist: Toggle Mode",
        "Voice: Say 'google'",
        "Canvas: 1 finger = Draw",
        "Canvas: 2 fingers = Erase",
        "Canvas: Pinch = Clear"
    ]
    if not canvas_mode:
        instructions = [
            "Fist: Toggle Mode",
            "Index: Move cursor",
            "Index+Thumb: Left click",
            "Middle+Thumb: Right click",
            "Index+Middle: Scroll"
        ]
    
    for i, text in enumerate(instructions[:5]):
        cv2.putText(frame, text, (10, frame.shape[0] - 120 + i*20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    # Show frames
    cv2.imshow("Voice-Controlled Virtual Mouse/Canvas", frame)
    
    if canvas_mode:
        # Add mode indicator to canvas
        canvas_display = canvas.copy()
        mode_indicator = "DRAW MODE" if drawing_active else "ERASE MODE" if erasing_active else "READY"
        cv2.putText(canvas_display, f"Mode: {mode_indicator}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(canvas_display, "Fist to exit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        cv2.imshow("Virtual Canvas", canvas_display)
    else:
        # Hide canvas window when in mouse mode
        cv2.destroyWindow("Virtual Canvas")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if left_click:
    pyautogui.mouseUp(button='left')
if right_click:
    pyautogui.mouseUp(button='right')
cap.release()
cv2.destroyAllWindows()