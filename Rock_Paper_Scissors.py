import cv2
import mediapipe as mp
import argparse
import os
import json

class GestureGame:
    def __init__(self, args):
        self.args = args
        if args.input:
            self.cap = cv2.VideoCapture(args.input)
        else:
            self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, landmarks):
        # Check for Rock (Hammer)
        if (landmarks[8].y > landmarks[6].y) and (landmarks[12].y > landmarks[10].y) and (landmarks[16].y > landmarks[14].y) and (landmarks[20].y > landmarks[18].y):
            return "Rock (Hammer)"
        
        # Check for Scissors
        if (landmarks[8].y < landmarks[6].y) and (landmarks[12].y < landmarks[10].y) and (landmarks[16].y > landmarks[14].y) and (landmarks[20].y > landmarks[18].y):
            return "Scissors"
        
        # Check for Paper
        if (landmarks[8].y < landmarks[6].y) and (landmarks[12].y < landmarks[10].y) and (landmarks[16].y < landmarks[14].y) and (landmarks[20].y < landmarks[18].y):
            return "Paper"
        
        return None

    def process_image(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get the hand landmarks
        results = self.hands.process(rgb_frame)
        
        json_output = {"hands": []}  # Initialize JSON structure
        
        # If hand landmarks are found, draw them and detect gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks.landmark)
                
                # Check if the hand is left or right
                handedness = "Left" if hand_landmarks.classification[0].label == "Left" else "Right"
                
                if gesture:
                    cv2.putText(frame, f"{handedness} Hand: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    json_output["hands"].append({"handedness": handedness, "gesture": gesture})
        
        # Save to JSON file if the --json argument is provided
        if self.args.json:
            with open('Rock_Paper_Scissors.json', 'w') as json_file:
                json.dump(json_output, json_file, indent=4)
        
        return frame

    def run(self):
        if self.args.input and self.args.input.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            # Process a single image
            ret, frame = self.cap.read()
            processed_frame = self.process_image(frame)
            if self.args.output:
                directory = os.path.dirname(self.args.output)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(self.args.output, processed_frame)
            if self.args.play:
                cv2.imshow('Rock Paper Scissors Game', processed_frame)
                cv2.waitKey(0)
        else:
            # Process video
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed_frame = self.process_image(frame)
                if self.args.play:
                    cv2.imshow('Rock Paper Scissors Game', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rock Paper Scissors Game with gestures.')
    parser.add_argument('-i', '--input', help='Path to the input image or video.')
    parser.add_argument('-o', '--output', help='Path to save the output image or video.')
    parser.add_argument('-n', '--no_image', action='store_true', help='Skip removing the image.')
    parser.add_argument('-j', '--json', action='store_true', help='Output the predicted parameter values as a Json file.')
    parser.add_argument('-p', '--play', action='store_true', help='Display the processed image or video.')
    args = parser.parse_args()

    game = GestureGame(args)
    game.run()
