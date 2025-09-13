import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from collections import deque

class HandCenterGestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Control settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # HAND CENTER DRAG SYSTEM
        self.pinch_start_time = None
        self.drag_threshold = 0.4
        self.is_dragging = False
        
        # Track HAND CENTER position for drag and movement
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        
        # Normal cursor control using hand center
        self.prev_hand_center = None
        self.sensitivity = 2.5
        
        # Other gestures
        self.last_action_time = 0
        self.action_cooldown = 0.15
        self.fist_history = deque(maxlen=8)
        
        # Two-finger scroll system (thumb position ignored)
        self.scroll_reference_y = None
        self.scroll_accumulated = 0
        self.scroll_exit_counter = 0
        
        print("🚀 HAND CENTER TRACKING CONTROLLER!")
        print("Uses whole hand center for natural movement")
        print("✋ Open hand → Move cursor (follows hand center)")
        print("🤏 Quick pinch → Left click")
        print("🤏 Hold pinch 0.4s → DRAG MODE (hand center controls drag)")
        print("✊ Fist → Right click")
        print("📱 Two fingers (index+middle) → Scroll up/down (thumb free!)")
        print("Press 'q' to quit")

    def get_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        return np.array(landmarks)

    def calculate_hand_center(self, landmarks):
        """Calculate the center of the hand using key landmarks"""
        # Use palm landmarks and wrist for a stable center point
        key_points = [
            landmarks[0],   # Wrist
            landmarks[5],   # Index MCP
            landmarks[9],   # Middle MCP
            landmarks[13],  # Ring MCP
            landmarks[17],  # Pinky MCP
        ]
        
        center_x = np.mean([point[0] for point in key_points])
        center_y = np.mean([point[1] for point in key_points])
        
        return [center_x, center_y]

    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def count_extended_fingers(self, landmarks):
        """Count extended fingers with LOOSER thresholds for better two-finger detection"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        extended_fingers = 0
        finger_states = []
        wrist = landmarks[0]
        
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # Thumb - make it harder to be "extended"
                thumb_wrist_dist = self.calculate_distance(landmarks[tip], wrist)
                thumb_pip_wrist_dist = self.calculate_distance(landmarks[pip], wrist)
                if thumb_wrist_dist > thumb_pip_wrist_dist + 0.03:  # Increased from 0.02
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)
            else:  # Other fingers - make it easier to be "extended"
                if landmarks[tip][1] < landmarks[pip][1] + 0.01:  # Reduced from 0.02
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)
        
        return extended_fingers, finger_states

    def detect_fist(self, landmarks):
        """Simple fist detection"""
        palm_center = landmarks[9]
        fingertip_distances = []
        for tip_idx in [4, 8, 12, 16, 20]:
            dist = self.calculate_distance(landmarks[tip_idx], palm_center)
            fingertip_distances.append(dist)
        
        avg_distance = np.mean(fingertip_distances)
        extended_count, _ = self.count_extended_fingers(landmarks)
        
        return extended_count <= 1 and avg_distance < 0.06

    def detect_open_hand(self, landmarks):
        """Simple open hand detection"""
        extended_count, _ = self.count_extended_fingers(landmarks)
        return extended_count >= 3

    def detect_two_finger_scroll(self, landmarks):
        """Detect two-finger scroll gesture - index and middle up, IGNORE thumb position"""
        extended_count, finger_states = self.count_extended_fingers(landmarks)
        
        # Check: index + middle up, ring + pinky down, IGNORE thumb completely
        is_two_finger_pose = (
            finger_states[1] and  # Index finger extended (MUST have)
            finger_states[2] and  # Middle finger extended (MUST have)
            not finger_states[3] and  # Ring finger not extended
            not finger_states[4]      # Pinky not extended
            # DON'T CARE about thumb (finger_states[0]) at all!
        )
        
        if not is_two_finger_pose:
            # Only reset if we've been out of pose for a bit (sticky mode)
            if not hasattr(self, 'scroll_exit_counter'):
                self.scroll_exit_counter = 0
            
            self.scroll_exit_counter += 1
            
            # Give it 3 frames of grace before exiting (prevents accidental exits)
            if self.scroll_exit_counter > 3:
                if self.scroll_reference_y is not None:
                    print("📱 Exiting scroll mode")
                self.scroll_reference_y = None
                self.scroll_accumulated = 0
                self.scroll_exit_counter = 0
            return False
        else:
            # Reset exit counter when back in pose
            self.scroll_exit_counter = 0
        
        # Calculate average Y position of index and middle fingertips
        index_tip_y = landmarks[8][1]  # Index fingertip
        middle_tip_y = landmarks[12][1]  # Middle fingertip
        current_fingers_y = (index_tip_y + middle_tip_y) / 2
        
        # Initialize reference position on first detection
        if self.scroll_reference_y is None:
            self.scroll_reference_y = current_fingers_y
            self.scroll_accumulated = 0
            print("📱 Two-finger scroll mode activated")
            return True
        
        # Calculate movement since reference
        movement = current_fingers_y - self.scroll_reference_y
        
        # Apply dead zone threshold (smaller movements ignored)
        dead_zone = 0.015  # Reduced from 0.02 - more sensitive
        if abs(movement) < dead_zone:
            return True  # In scroll mode but not moving enough
        
        # Accumulate movement for bigger-movement requirement
        self.scroll_accumulated += movement
        
        # Require larger accumulated movement before scrolling
        scroll_threshold = 0.035  # Reduced from 0.05 - easier to trigger
        
        if abs(self.scroll_accumulated) >= scroll_threshold:
            # Determine scroll direction and amount
            if self.scroll_accumulated > 0:
                # Fingers moved down -> scroll down
                scroll_amount = -2  # Negative = scroll down
                direction = "down"
            else:
                # Fingers moved up -> scroll up
                scroll_amount = 2   # Positive = scroll up
                direction = "up"
            
            # Perform the scroll
            pyautogui.scroll(scroll_amount)
            print(f"📜 Two-finger scroll {direction} (movement: {self.scroll_accumulated:.3f})")
            
            # Reset accumulator but keep reference for continuous scrolling
            self.scroll_accumulated = 0
            self.scroll_reference_y = current_fingers_y
        
        return True

    def detect_gestures(self, landmarks):
        """Gesture detection using HAND CENTER tracking"""
        current_time = time.time()
        
        # Calculate hand center
        hand_center = self.calculate_hand_center(landmarks)
        
        # Get key finger positions for pinch detection
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate pinch
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        is_pinched = pinch_distance < 0.05
        
        extended_count, finger_states = self.count_extended_fingers(landmarks)
        
        # 1. FIST DETECTION for RIGHT CLICK
        is_fist = self.detect_fist(landmarks)
        self.fist_history.append(is_fist)
        
        if len(self.fist_history) >= 5:
            recent_states = list(self.fist_history)[-5:]
            if (not recent_states[-1] and not recent_states[-2] and
                any(recent_states[:-2]) and
                current_time - self.last_action_time > self.action_cooldown):
                
                pyautogui.rightClick()
                self.last_action_time = current_time
                return "fist_right_click"
        
        # 2. PINCH/DRAG SYSTEM using HAND CENTER
        if is_pinched:
            # Start timing pinch
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time
                print("🤏 Pinch started - timing...")
            
            pinch_duration = current_time - self.pinch_start_time
            
            # Start drag after threshold
            if pinch_duration >= self.drag_threshold and not self.is_dragging:
                try:
                    # Get current screen position
                    current_screen_x, current_screen_y = pyautogui.position()
                    
                    # Store HAND CENTER position at drag start
                    self.drag_start_hand_pos = hand_center.copy()
                    self.drag_start_screen_pos = [current_screen_x, current_screen_y]
                    
                    # Start drag
                    pyautogui.mouseDown(button='left')
                    self.is_dragging = True
                    self.last_action_time = current_time
                    
                    print(f"🖱️ DRAG STARTED! Hand center at ({hand_center[0]:.4f}, {hand_center[1]:.4f})")
                    print(f"🖱️ Screen position: ({current_screen_x}, {current_screen_y})")
                    
                    return "drag_started"
                    
                except Exception as e:
                    print(f"❌ Drag start failed: {e}")
                    return "drag_failed"
            
            # Continue dragging - track HAND CENTER movement
            if self.is_dragging and self.drag_start_hand_pos is not None:
                # Calculate how much the HAND CENTER has moved since drag started
                hand_delta_x = hand_center[0] - self.drag_start_hand_pos[0]
                hand_delta_y = hand_center[1] - self.drag_start_hand_pos[1]
                
                # Convert hand movement to screen movement
                screen_delta_x = hand_delta_x * self.screen_width * 3.0
                screen_delta_y = hand_delta_y * self.screen_height * 3.0
                
                # Calculate new screen position
                new_screen_x = self.drag_start_screen_pos[0] + screen_delta_x
                new_screen_y = self.drag_start_screen_pos[1] + screen_delta_y
                
                # Clamp to screen bounds
                new_screen_x = max(20, min(self.screen_width - 20, new_screen_x))
                new_screen_y = max(20, min(self.screen_height - 20, new_screen_y))
                
                try:
                    # Move cursor to new position
                    pyautogui.moveTo(new_screen_x, new_screen_y, duration=0)
                    
                    # Check actual movement
                    actual_x, actual_y = pyautogui.position()
                    total_moved = abs(actual_x - self.drag_start_screen_pos[0]) + abs(actual_y - self.drag_start_screen_pos[1])
                    
                    if total_moved > 5:  # Only print if significant movement
                        print(f"🖱️ Dragging: hand Δ({hand_delta_x:.4f},{hand_delta_y:.4f}) → screen ({actual_x},{actual_y}) [moved {total_moved:.0f}px]")
                        
                except Exception as e:
                    print(f"❌ Drag move failed: {e}")
                
                return "dragging"
            
            # Waiting for drag threshold
            remaining = self.drag_threshold - pinch_duration
            if remaining > 0:
                return f"pinch_waiting_{remaining:.1f}"
        
        else:
            # Pinch released
            if self.pinch_start_time is not None:
                pinch_duration = current_time - self.pinch_start_time
                
                # End drag if was dragging
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')
                        
                        # Calculate total drag distance
                        if self.drag_start_screen_pos is not None:
                            final_x, final_y = pyautogui.position()
                            total_distance = abs(final_x - self.drag_start_screen_pos[0]) + abs(final_y - self.drag_start_screen_pos[1])
                            print(f"🖱️ DRAG ENDED! Total distance: {total_distance} pixels")
                        
                        self.is_dragging = False
                        self.drag_start_hand_pos = None
                        self.drag_start_screen_pos = None
                        self.pinch_start_time = None
                        
                        return "drag_ended"
                        
                    except Exception as e:
                        print(f"❌ Drag end failed: {e}")
                        self.is_dragging = False
                        return "drag_end_failed"
                
                # Quick click if short pinch
                elif (pinch_duration < self.drag_threshold and
                      current_time - self.last_action_time > self.action_cooldown):
                    
                    try:
                        pyautogui.click()
                        self.last_action_time = current_time
                        print("🖱️ CLICK!")
                        self.pinch_start_time = None
                        return "pinch_click"
                    except Exception as e:
                        print(f"❌ Click failed: {e}")
                        return "click_failed"
                
                # Clean up
                self.pinch_start_time = None
        
        # 3. TWO-FINGER SCROLL DETECTION
        if not is_pinched and not self.is_dragging:
            if self.detect_two_finger_scroll(landmarks):
                if self.scroll_reference_y is not None:
                    return "two_finger_scroll"
        
        # 4. NORMAL CURSOR CONTROL using HAND CENTER
        if not is_pinched and not self.is_dragging and self.detect_open_hand(landmarks):
            
            # Don't control cursor if in scroll mode
            if self.scroll_reference_y is not None:
                return "scroll_mode_active"
            
            # Track HAND CENTER movement for cursor control
            if self.prev_hand_center is not None:
                hand_delta_x = hand_center[0] - self.prev_hand_center[0]
                hand_delta_y = hand_center[1] - self.prev_hand_center[1]
                
                # Move cursor based on hand center movement
                if abs(hand_delta_x) > 0.001 or abs(hand_delta_y) > 0.001:
                    screen_delta_x = hand_delta_x * self.screen_width * self.sensitivity
                    screen_delta_y = hand_delta_y * self.screen_height * self.sensitivity
                    
                    try:
                        current_x, current_y = pyautogui.position()
                        new_x = max(20, min(self.screen_width - 20, current_x + screen_delta_x))
                        new_y = max(20, min(self.screen_height - 20, current_y + screen_delta_y))
                        
                        pyautogui.moveTo(new_x, new_y, duration=0)
                        
                    except Exception as e:
                        print(f"❌ Cursor move failed: {e}")
            
            # Update previous HAND CENTER position
            self.prev_hand_center = hand_center.copy()
            return "cursor_control"
        
        else:
            # Reset hand center tracking when not controlling cursor (unless dragging)
            if not self.is_dragging:
                self.prev_hand_center = None
        
        return "idle"

    def draw_debug_info(self, frame, landmarks, gesture, extended_count, finger_states):
        """Debug visualization focusing on HAND CENTER tracking and two-finger scroll"""
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate hand center
        hand_center = self.calculate_hand_center(landmarks)
        
        # Get finger positions for pinch visualization
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        is_pinched = pinch_distance < 0.05
        
        # Clean background
        cv2.rectangle(frame, (10, 10), (600, 240), (0, 0, 0), -1)
        
        # Current gesture
        color = (0, 255, 0) if gesture != "idle" else (255, 255, 255)
        if "drag" in gesture:
            color = (255, 0, 128)  # Pink for drag
        elif "pinch_waiting" in gesture:
            color = (255, 128, 0)  # Orange for waiting
        elif "two_finger_scroll" in gesture or "scroll_mode" in gesture:
            color = (0, 255, 255)  # Cyan for scroll
        
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # HAND CENTER focus
        cv2.putText(frame, f"Hand Center: ({hand_center[0]:.4f}, {hand_center[1]:.4f})", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Two-finger scroll status
        if self.scroll_reference_y is not None:
            cv2.putText(frame, f"📱 SCROLL MODE - Move 2 fingers up/down", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ref Y: {self.scroll_reference_y:.4f}, Acc: {self.scroll_accumulated:.3f}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Drag state
        elif self.is_dragging:
            cv2.putText(frame, "🖱️ ACTIVELY DRAGGING!", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 128), 2)
            if self.drag_start_hand_pos is not None:
                start_x, start_y = self.drag_start_hand_pos
                cv2.putText(frame, f"Drag start: ({start_x:.4f}, {start_y:.4f})", (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif self.pinch_start_time is not None:
            remaining = max(0, self.drag_threshold - (time.time() - self.pinch_start_time))
            if remaining > 0:
                cv2.putText(frame, f"Hold {remaining:.1f}s more for drag", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Extended fingers info
        is_open_hand = self.detect_open_hand(landmarks)
        cv2.putText(frame, f"Extended fingers: {extended_count}/5", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show finger states for two-finger detection (highlight thumb ignorance)
        finger_names = ["Thumb*", "Index", "Middle", "Ring", "Pinky"]
        finger_colors = [(150, 150, 150), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
        finger_text = ""
        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
            status = "✓" if state else "✗"
            if i == 0:  # Thumb
                finger_text += f"{name}: {status}(ignored) | "
            else:
                finger_text += f"{name}: {status} | "
        
        cv2.putText(frame, finger_text[:-3], (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        open_hand_color = (0, 255, 0) if is_open_hand else (255, 0, 0)
        cv2.putText(frame, f"Open hand: {'YES' if is_open_hand else 'NO'}", (20, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, open_hand_color, 2)
        
        # Instructions
        cv2.putText(frame, "Scroll: Index+Middle up, Ring+Pinky down (thumb free!)", (20, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        
        # Visual indicators
        hand_center_pos = (int(hand_center[0] * frame_width), int(hand_center[1] * frame_height))
        thumb_pos = (int(thumb_tip[0] * frame_width), int(thumb_tip[1] * frame_height))
        index_pos = (int(index_tip[0] * frame_width), int(index_tip[1] * frame_height))
        middle_pos = (int(middle_tip[0] * frame_width), int(middle_tip[1] * frame_height))
        
        # Pinch line
        line_color = (255, 0, 128) if self.is_dragging else (0, 255, 0) if is_pinched else (255, 255, 255)
        line_thickness = 6 if self.is_dragging else 3 if is_pinched else 2
        cv2.line(frame, thumb_pos, index_pos, line_color, line_thickness)
        
        # Two-finger scroll indicators
        if self.scroll_reference_y is not None:
            # Highlight the two scroll fingers
            cv2.circle(frame, index_pos, 15, (0, 255, 255), 4)
            cv2.circle(frame, middle_pos, 15, (0, 255, 255), 4)
            cv2.line(frame, index_pos, middle_pos, (0, 255, 255), 4)
            cv2.putText(frame, "SCROLL", (index_pos[0] + 20, index_pos[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Highlight HAND CENTER as the main tracking point
        center_color = (255, 0, 128) if self.is_dragging else (0, 255, 255) if self.scroll_reference_y else (0, 255, 255)
        center_size = 20 if self.is_dragging else 15
        cv2.circle(frame, hand_center_pos, center_size, center_color, 4)
        cv2.putText(frame, "HAND CENTER", (hand_center_pos[0] + 25, hand_center_pos[1] - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_color, 2)
        
        # Show drag start position if dragging
        if self.is_dragging and self.drag_start_hand_pos is not None:
            start_screen_x = int(self.drag_start_hand_pos[0] * frame_width)
            start_screen_y = int(self.drag_start_hand_pos[1] * frame_height)
            cv2.circle(frame, (start_screen_x, start_screen_y), 12, (255, 255, 0), 3)
            cv2.line(frame, (start_screen_x, start_screen_y), hand_center_pos, (255, 255, 0), 3)
            cv2.putText(frame, "START", (start_screen_x + 15, start_screen_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

    def run(self):
        """Main control loop"""
        print("🚀 Starting HAND CENTER tracking controller with two-finger scroll...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            gesture = "no_hand"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    
                    landmarks = self.get_landmarks(hand_landmarks)
                    extended_count, finger_states = self.count_extended_fingers(landmarks)
                    gesture = self.detect_gestures(landmarks)
                    
                    self.draw_debug_info(frame, landmarks, gesture, extended_count, finger_states)
            else:
                # Clean up when hand lost
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')
                        print("🖱️ EMERGENCY: Hand lost during drag - ended drag")
                    except:
                        pass
                    self.is_dragging = False
                
                self.pinch_start_time = None
                self.drag_start_hand_pos = None
                self.drag_start_screen_pos = None
                self.prev_hand_center = None
                self.scroll_reference_y = None
                self.scroll_accumulated = 0
                self.scroll_exit_counter = 0
                
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
                cv2.putText(frame, "Show your hand", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Hand Center Controller', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Final cleanup
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button='left')
            except:
                pass
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Hand center controller stopped")

if __name__ == "__main__":
    try:
        controller = HandCenterGestureController()
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
