import cv2
import mediapipe as mp
import argparse
import os
import json

class FingerCounter:
    def __init__(self, input_path, output_path, no_image, json_output, play):
        self.input_path = input_path
        self.output_path = output_path
        self.no_image = no_image
        self.json_output = json_output
        self.play = play
        self.results_list = []

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        # Determine if input is an image or video
        if input_path.endswith('.jpg') or input_path.endswith('.png'):
            self.input_type = 'image'
            self.image = cv2.imread(input_path)
        else:
            self.input_type = 'video'
            self.cap = cv2.VideoCapture(input_path)

    def count_fingers(self, landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        count = 0

        # Determine if it's a left or right hand
        if landmarks[17].x > landmarks[5].x:
            hand_type = "Right"
        else:
            hand_type = "Left"

        # Thumb
        if hand_type == "Right":
            if landmarks[4].x < landmarks[3].x and landmarks[4].y < landmarks[3].y:
                count += 1
        else:
            if landmarks[4].x > landmarks[3].x and landmarks[4].y < landmarks[3].y:
                count += 1

        # Other fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_tips[i]-2].y:
                count += 1

        return count

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
    
        left_hand_count = None
        right_hand_count = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                finger_count = self.count_fingers(hand_landmarks.landmark)
            
                hand_type = "Right" if hand_landmarks.landmark[17].x > hand_landmarks.landmark[5].x else "Left"
            
                if hand_type == "Left":
                    left_hand_count = finger_count
                else:
                    right_hand_count = finger_count

                # Display finger count on the frame
                display_text = str(finger_count) if finger_count > 0 else "nothing"
                color = (0, 0, 255) if hand_type == "Right" else (255, 0, 0)
                x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, display_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

        # Add results to results_list for JSON output
        output = {
            "Number_of_fingers_left": left_hand_count,
            "Number_of_fingers_right": right_hand_count,
            "Hand_detected": bool(results.multi_hand_landmarks)
        }
        self.results_list.append(output)

        return frame


    def run(self):
        if self.input_type == 'image':
            processed_image = self.process_frame(self.image)
            if self.play:
                cv2.imshow('Finger Counter', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if self.output_path:
                cv2.imwrite(self.output_path, processed_image)
            if self.json_output:
                with open('output.json', 'w') as f:
                    json.dump(self.results_list[0], f)
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame)
                if self.play:
                    cv2.imshow('Finger Counter', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
            if self.json_output:
                with open('output.json', 'w') as f:
                    json.dump(self.results_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finger Counter using MediaPipe')
    parser.add_argument('-i', '--input', required=True, help='Path to input image or video')
    parser.add_argument('-o', '--output', help='Path to save processed image or video')
    parser.add_argument('-n', '--no_image', action='store_true', help='Skip removing the image')
    parser.add_argument('-j', '--json', action='store_true', help='Output predicted parameter values as a JSON file')
    parser.add_argument('-p', '--play', action='store_true', help='Display the processed image or video')

    args = parser.parse_args()

    # Check if output directory exists, if not create it
    if args.output:
        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    fc = FingerCounter(args.input, args.output, args.no_image, args.json, args.play)
    fc.run()
