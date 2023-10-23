import cv2
import mediapipe as mp
import argparse
import os
import json

class HandRaiseDetection:
    def __init__(self, input_path, output_path, no_image, json_output, play):
        self.input_path = input_path
        self.output_path = output_path
        self.no_image = no_image
        self.json_output = json_output
        self.play = play

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def is_hand_raised(self, landmarks):
        # Check if the wrist's y-coordinate is above a certain threshold
        if landmarks[0].y < 0.5:  # Adjust the threshold value as needed
            return True
        return False

    def process_image(self, image):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame and get the hand landmarks
        results = self.hands.process(rgb_frame)

        raised_hands = []

        # If hand landmarks are found, draw them and check if hand is raised
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.is_hand_raised(hand_landmarks.landmark):
                    label = "Left" if hand_info.classification[0].label == "Left" else "Right"
                    raised_hands.append(label)

        if "Left" in raised_hands and "Right" in raised_hands:
            message = "Both hands"
        elif "Left" in raised_hands:
            message = "Raised left hand"
        elif "Right" in raised_hands:
            message = "Raised right hand"
        else:
            message = ""

        cv2.putText(image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return image, results

    def run(self):
        if self.input_path.endswith('.jpg') or self.input_path.endswith('.png'):
            # Process image
            image = cv2.imread(self.input_path)
            processed_image, results = self.process_image(image)
            if self.play:
                cv2.imshow('Hand Raise Detection', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if not self.no_image:
                cv2.imwrite(self.output_path, processed_image)
            if self.json_output:
                # Save results to JSON (modify as needed)
                output_data = {
                    # Add your desired JSON output format here
                }
                with open(self.json_output, 'w') as json_file:
                    json.dump(output_data, json_file)
        else:
            # Process video
            cap = cv2.VideoCapture(self.input_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, results = self.process_image(frame)
                if self.play:
                    cv2.imshow('Hand Raise Detection', processed_frame)
                out.write(processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Raise Detection')
    parser.add_argument('-i', '--input', required=True, help='Path to input image/video')
    parser.add_argument('-o', '--output', required=True, help='Path to save output image/video')
    parser.add_argument('-n', '--no_image', action='store_true', help='Skip saving the image')
    parser.add_argument('-j', '--json', help='Path to save the JSON output')
    parser.add_argument('-p', '--play', action='store_true', help='Display the processed image/video')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hrd = HandRaiseDetection(args.input, args.output, args.no_image, args.json, args.play)
    hrd.run()
