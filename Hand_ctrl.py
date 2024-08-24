import cv2
import mediapipe as mp
import rsautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbc

rsautogui.FAILSAFE = False
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

class GestureType(IntEnum):
    CLOSED = 0
    LITTLE = 1
    FOURTH = 2
    MIDDLE = 4
    LAST_THREE = 7
    POINTER = 8
    FIRST_TWO = 12
    LAST_FOUR = 15
    THUMB = 16    
    OPEN = 31
    
    VICTORY = 33
    TWO_CLOSED = 34
    PINCH_MAIN = 35
    PINCH_SECONDARY = 36

class HandType(IntEnum):
    SECONDARY = 0
    PRIMARY = 1

class GestureRecognizer:
    def __init__(self, hand_type):
        self.digit = 0
        self.initial_gesture = GestureType.OPEN
        self.last_gesture = GestureType.OPEN
        self.frame_count = 0
        self.hand_data = None
        self.hand_type = hand_type
    
    def update_hand_data(self, hand_data):
        self.hand_data = hand_data

    def calculate_distance(self, point_a, point_b):
        return math.sqrt(
            (self.hand_data.landmark[point_a].x - self.hand_data.landmark[point_b].x)**2 +
            (self.hand_data.landmark[point_a].y - self.hand_data.landmark[point_b].y)**2
        )
    
    def calculate_signed_distance(self, point_a, point_b):
        sign = 1 if self.hand_data.landmark[point_a].y < self.hand_data.landmark[point_b].y else -1
        return sign * self.calculate_distance([point_a, point_b])
    
    def calculate_depth(self, point_a, point_b):
        return abs(self.hand_data.landmark[point_a].z - self.hand_data.landmark[point_b].z)
    
    def set_digit_state(self):
        if self.hand_data is None:
            return

        finger_points = [[8,5,0], [12,9,0], [16,13,0], [20,17,0]]
        self.digit = 0
        for idx, point in enumerate(finger_points):
            dist_tip_mid = self.calculate_signed_distance(point[0], point[1])
            dist_mid_base = self.calculate_signed_distance(point[1], point[2])
            
            ratio = round(dist_tip_mid / (dist_mid_base or 0.01), 1)

            self.digit = self.digit << 1
            if ratio > 0.5:
                self.digit = self.digit | 1
    
    def recognize_gesture(self):
        if self.hand_data is None:
            return GestureType.OPEN

        current_gesture = GestureType.OPEN
        if self.digit in [GestureType.LAST_THREE, GestureType.LAST_FOUR] and self.calculate_distance([8,4]) < 0.05:
            current_gesture = GestureType.PINCH_SECONDARY if self.hand_type == HandType.SECONDARY else GestureType.PINCH_MAIN
        elif GestureType.FIRST_TWO == self.digit:
            dist_fingertips = self.calculate_distance([8,12])
            dist_knuckles = self.calculate_distance([5,9])
            if dist_fingertips / dist_knuckles > 1.7:
                current_gesture = GestureType.VICTORY
            elif self.calculate_depth([8,12]) < 0.1:
                current_gesture = GestureType.TWO_CLOSED
            else:
                current_gesture = GestureType.MIDDLE
        else:
            current_gesture = self.digit
        
        if current_gesture == self.last_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.last_gesture = current_gesture

        if self.frame_count > 4:
            self.initial_gesture = current_gesture
        return self.initial_gesture

class InputController:
    prev_x, prev_y = 0, 0
    is_v_gesture = False
    is_fist = False
    is_pinch_main = False
    is_pinch_secondary = False
    pinch_start_x, pinch_start_y = None, None
    pinch_direction = None
    prev_pinch_level = 0
    pinch_level = 0
    frame_count = 0
    prev_hand_position = None
    pinch_threshold = 0.3
    
    @staticmethod
    def get_pinch_level_y(hand_data):
        return round((InputController.pinch_start_y - hand_data.landmark[8].y) * 10, 1)

    @staticmethod
    def get_pinch_level_x(hand_data):
        return round((hand_data.landmark[8].x - InputController.pinch_start_x) * 10, 1)
    
    @staticmethod
    def adjust_brightness():
        current_brightness = sbc.get_brightness(display=0) / 100.0
        new_brightness = max(0.0, min(1.0, current_brightness + InputController.pinch_level / 50.0))
        sbc.fade_brightness(int(100 * new_brightness), start=sbc.get_brightness(display=0))
    
    @staticmethod
    def adjust_volume():
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current_volume = volume.GetMasterVolumeLevelScalar()
        new_volume = max(0.0, min(1.0, current_volume + InputController.pinch_level / 50.0))
        volume.SetMasterVolumeLevelScalar(new_volume, None)
    
    @staticmethod
    def scroll_vertical():
        rsautogui.scroll(120 if InputController.pinch_level > 0.0 else -120)
    
    @staticmethod
    def scroll_horizontal():
        rsautogui.keyDown('shift')
        rsautogui.keyDown('ctrl')
        rsautogui.scroll(-120 if InputController.pinch_level > 0.0 else 120)
        rsautogui.keyUp('ctrl')
        rsautogui.keyUp('shift')

    @staticmethod
    def get_cursor_position(hand_data):
        point = 9
        screen_width, screen_height = rsautogui.size()
        x = int(hand_data.landmark[point].x * screen_width)
        y = int(hand_data.landmark[point].y * screen_height)

        if InputController.prev_hand_position is None:
            InputController.prev_hand_position = x, y

        delta_x = x - InputController.prev_hand_position[0]
        delta_y = y - InputController.prev_hand_position[1]

        distance_sq = delta_x**2 + delta_y**2
        smoothing_factor = 0 if distance_sq <= 25 else 0.07 * (distance_sq ** 0.5) if distance_sq <= 900 else 2.1

        current_x, current_y = rsautogui.position()
        new_x = current_x + delta_x * smoothing_factor
        new_y = current_y + delta_y * smoothing_factor

        InputController.prev_hand_position = [x, y]
        return (new_x, new_y)

    @staticmethod
    def initialize_pinch(hand_data):
        InputController.pinch_start_x = hand_data.landmark[8].x
        InputController.pinch_start_y = hand_data.landmark[8].y
        InputController.pinch_level = 0
        InputController.prev_pinch_level = 0
        InputController.frame_count = 0

    @staticmethod
    def handle_pinch(hand_data, horizontal_action, vertical_action):
        if InputController.frame_count == 5:
            InputController.frame_count = 0
            InputController.pinch_level = InputController.prev_pinch_level

            if InputController.pinch_direction:
                horizontal_action()
            else:
                vertical_action()

        level_x = InputController.get_pinch_level_x(hand_data)
        level_y = InputController.get_pinch_level_y(hand_data)
            
        if abs(level_y) > abs(level_x) and abs(level_y) > InputController.pinch_threshold:
            InputController.pinch_direction = False
            if abs(InputController.prev_pinch_level - level_y) < InputController.pinch_threshold:
                InputController.frame_count += 1
            else:
                InputController.prev_pinch_level = level_y
                InputController.frame_count = 0

        elif abs(level_x) > InputController.pinch_threshold:
            InputController.pinch_direction = True
            if abs(InputController.prev_pinch_level - level_x) < InputController.pinch_threshold:
                InputController.frame_count += 1
            else:
                InputController.prev_pinch_level = level_x
                InputController.frame_count = 0

    @staticmethod
    def process_gesture(gesture, hand_data):
        cursor_x, cursor_y = None, None
        if gesture != GestureType.OPEN:
            cursor_x, cursor_y = InputController.get_cursor_position(hand_data)
        
        if gesture != GestureType.CLOSED and InputController.is_fist:
            InputController.is_fist = False
            rsautogui.mouseUp(button="left")

        if gesture != GestureType.PINCH_MAIN and InputController.is_pinch_main:
            InputController.is_pinch_main = False

        if gesture != GestureType.PINCH_SECONDARY and InputController.is_pinch_secondary:
            InputController.is_pinch_secondary = False

        if gesture == GestureType.VICTORY:
            InputController.is_v_gesture = True
            rsautogui.moveTo(cursor_x, cursor_y, duration=0.1)

        elif gesture == GestureType.CLOSED:
            if not InputController.is_fist:
                InputController.is_fist = True
                rsautogui.mouseDown(button="left")
            rsautogui.moveTo(cursor_x, cursor_y, duration=0.1)

        elif gesture == GestureType.MIDDLE and InputController.is_v_gesture:
            rsautogui.click()
            InputController.is_v_gesture = False

        elif gesture == GestureType.POINTER and InputController.is_v_gesture:
            rsautogui.click(button='right')
            InputController.is_v_gesture = False

        elif gesture == GestureType.TWO_CLOSED and InputController.is_v_gesture:
            rsautogui.doubleClick()
            InputController.is_v_gesture = False

        elif gesture == GestureType.PINCH_SECONDARY:
            if not InputController.is_pinch_secondary:
                InputController.initialize_pinch(hand_data)
                InputController.is_pinch_secondary = True
            InputController.handle_pinch(hand_data, InputController.scroll_horizontal, InputController.scroll_vertical)
        
        elif gesture == GestureType.PINCH_MAIN:
            if not InputController.is_pinch_main:
                InputController.initialize_pinch(hand_data)
                InputController.is_pinch_main = True
            InputController.handle_pinch(hand_data, InputController.adjust_brightness, InputController.adjust_volume)

class GestureControlMain:
    is_active = 0
    video_capture = None
    FRAME_HEIGHT = None
    FRAME_WIDTH = None
    primary_hand = None
    secondary_hand = None
    is_right_dominant = True

    def __init__(self):
        GestureControlMain.is_active = 1
        GestureControlMain.video_capture = cv2.VideoCapture(0)
        GestureControlMain.FRAME_HEIGHT = GestureControlMain.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureControlMain.FRAME_WIDTH = GestureControlMain.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    @staticmethod
    def categorize_hands(results):
        left, right = None, None
        try:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                if handedness_dict['classification'][0]['label'] == 'Right':
                    right = results.multi_hand_landmarks[idx]
                else:
                    left = results.multi_hand_landmarks[idx]
        except:
            pass
        
        GestureControlMain.primary_hand = right if GestureControlMain.is_right_dominant else left
        GestureControlMain.secondary_hand = left if GestureControlMain.is_right_dominant else right

    def run(self):
        primary_recognizer = GestureRecognizer(HandType.PRIMARY)
        secondary_recognizer = GestureRecognizer(HandType.SECONDARY)

        with mp_hand.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureControlMain.video_capture.isOpened() and GestureControlMain.is_active:
                success, frame = GestureControlMain.video_capture.read()

                if not success:
                    print("Failed to capture frame.")
                    continue
                
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = hands.process(frame)
                
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:                   
                    GestureControlMain.categorize_hands(results)
                    primary_recognizer.update_hand_data(GestureControlMain.primary_hand)
                    secondary_recognizer.update_hand_data(GestureControlMain.secondary_hand)

                    primary_recognizer.set_digit_state()
                    secondary_recognizer.set_digit_state()
                    gesture = secondary_recognizer.recognize_gesture()

                    if gesture == GestureType.PINCH_SECONDARY:
                        InputController.process_gesture(gesture, secondary_recognizer.hand_data)
                    else:
                        gesture = primary_recognizer.recognize_gesture()
                        InputController.process_gesture(gesture, primary_recognizer.hand_data)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)
                else:
                    InputController.prev_hand_position = None
                cv2.imshow('Gesture Control Interface', frame)
                if cv2.waitKey(5) & 0xFF == 13:  # Enter key
                    break
        GestureControlMain.video_capture.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    gesture_control = GestureControlMain()
    gesture_control.run()
