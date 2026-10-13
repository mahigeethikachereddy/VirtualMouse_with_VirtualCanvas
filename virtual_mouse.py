import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Constants
PINCH_THRESHOLD = 0.05  # Distance threshold for clicks
SCROLL_ACTIVATION_THRESHOLD = 0.1  # Finger distance for scroll
SCROLL_DEADZONE = 5  # Minimum movement to trigger scroll
SCROLL_SMOOTHING = 5  # Frames to average for smoothing
SCROLL_SPEED = 0.3  # Reduced scroll speed multiplier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Camera setup
cap = cv2.VideoCapture(0)

# State variables
prev_x, prev_y = 0, 0
left_click = False
right_click = False
scroll_active = False
scroll_buffer = []
scroll_reference_y = 0

def is_scroll_gesture(hand_landmarks):
    """Check if only index and middle fingers are extended and close together"""
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    
    # Check if other fingers are down
    if (ring_tip.y < hand_landmarks.landmark[13].y or  # Ring finger up
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[17].y):  # Pinky up
        return False
        
    # Check distance between index and middle fingers
    distance = np.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
    return distance < SCROLL_ACTIVATION_THRESHOLD

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get landmarks
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        # Convert to screen coordinates
        curr_x = int(index_tip.x * screen_w)
        curr_y = int(index_tip.y * screen_h)
        
        # Smooth cursor movement
        smoothed_x = prev_x + (curr_x - prev_x) * 0.2
        smoothed_y = prev_y + (curr_y - prev_y) * 0.2
        pyautogui.moveTo(smoothed_x, smoothed_y)
        prev_x, prev_y = smoothed_x, smoothed_y

        # Check for scroll gesture
        if is_scroll_gesture(hand_landmarks):
            if not scroll_active:
                scroll_active = True
                scroll_reference_y = index_tip.y
                scroll_buffer = []
            
            # Calculate vertical movement
            delta_y = (index_tip.y - scroll_reference_y) * screen_h
            scroll_buffer.append(delta_y)
            
            if len(scroll_buffer) > SCROLL_SMOOTHING:
                scroll_buffer.pop(0)
            
            if scroll_buffer and abs(delta_y) > SCROLL_DEADZONE:
                scroll_amount = -delta_y * SCROLL_SPEED  # Reduced speed
                pyautogui.scroll(int(scroll_amount))
                scroll_reference_y = index_tip.y  # Reset reference
        else:
            scroll_active = False
            
            # Check for left click (index + thumb)
            left_pinch = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            if left_pinch < PINCH_THRESHOLD:
                if not left_click:
                    pyautogui.mouseDown(button='left')
                    left_click = True
            elif left_click:
                pyautogui.mouseUp(button='left')
                left_click = False
            
            # Check for right click (middle + thumb)
            right_pinch = np.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
            if right_pinch < PINCH_THRESHOLD:
                if not right_click:
                    pyautogui.mouseDown(button='right')
                    right_click = True
            elif right_click:
                pyautogui.mouseUp(button='right')
                right_click = False

        # Draw hand landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display status
    if left_click:
        cv2.putText(frame, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif right_click:
        cv2.putText(frame, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if scroll_active:
        cv2.putText(frame, "SCROLLING", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Simple Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if left_click:
    pyautogui.mouseUp(button='left')
if right_click:
    pyautogui.mouseUp(button='right')
cap.release()
cv2.destroyAllWindows()