import cv2
import mediapipe as mp
import argparse
import os
import json

class GestureRecognition:
    def __init__(self, input_path, output_path, play, no_image, json_output, threshold):
        if input_path.endswith('.png') or input_path.endswith('.jpg'):
            self.cap = cv2.imread(input_path)
            self.is_image = True
        else:
            self.cap = cv2.VideoCapture(input_path)
            self.is_image = False
        self.output_path = output_path
        self.play = play
        self.no_image = no_image
        self.json_output = json_output
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.threshold = threshold

    def detect_gesture(self, landmarks):
        # Check direction of index finger
        dx = landmarks[8].x - landmarks[5].x
        dy = landmarks[8].y - landmarks[5].y

        # Use the provided threshold
        threshold = self.threshold

        # Check for "nothing" gesture where all fingers are upside down
        if landmarks[4].y > landmarks[3].y and \
            landmarks[8].y > landmarks[6].y and \
            landmarks[12].y > landmarks[10].y and \
            landmarks[16].y > landmarks[14].y and \
            landmarks[20].y > landmarks[18].y:  # All fingers are upside down
            return "nothing"
        
        elif abs(dx) > threshold and dx > abs(dy):
            return "Index pointing right"
        elif abs(dx) > threshold and dx < -abs(dy):
            return "Index pointing left"
        elif abs(dy) > threshold and dy > abs(dx):
            return "Index pointing down"
        elif abs(dy) > threshold and dy < -abs(dx):
            return "Index pointing up"
        
        # Check for "bad" gesture using only middle finger
        elif landmarks[12].y < landmarks[10].y and \
            landmarks[8].y > landmarks[6].y and \
            landmarks[16].y > landmarks[14].y and \
            landmarks[20].y > landmarks[18].y:  # Only middle finger is relatively extended
            return "bad"
        # Check for "Hold up two fingers" gesture using index and middle fingers
        elif landmarks[8].y < landmarks[6].y - 0.02 and \
            landmarks[12].y < landmarks[10].y - 0.02 and \
            landmarks[16].y > landmarks[14].y + 0.02 and \
            landmarks[20].y > landmarks[18].y + 0.02:  # Only index and middle fingers are extended
            return "Hold up two fingers"
        # Check for "ok" gesture using thumb and index
        elif self.distance(landmarks[4], landmarks[8]) < 0.05 and \
            landmarks[16].y > landmarks[14].y:  # Ensure ring finger is not extended
            return "promise"
        # Check for "like" gesture using thumb and index
        elif landmarks[4].y < landmarks[3].y and \
            landmarks[16].y > landmarks[14].y:  # Ensure ring finger is not extended
            return "like"
        # Check for "dislike" gesture using thumb and index
        elif landmarks[4].y > landmarks[3].y - 0.03 and \
            landmarks[16].y > landmarks[14].y - 0.03:  # Ensure ring finger is not extended
            return "dislike"
        # Check for "promise" gesture using only ring finger
        elif landmarks[16].y < landmarks[14].y:  # Only ring finger is extended
            return "ok"
        return None
    

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5

    def run(self):
        if self.is_image:
            frame = self.cap
            self.process_frame(frame)
            if self.play:
                cv2.imshow('Gesture Recognition', frame)
                cv2.waitKey(0)
            if not self.no_image:
                cv2.imwrite(self.output_path, frame)
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.process_frame(frame)
                if self.play:
                    cv2.imshow('Gesture Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()

    def compute_bounding_box(self, landmarks):
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        box_cx = sum(x_coords) / len(landmarks)
        box_cy = sum(y_coords) / len(landmarks)
        box_w = max(x_coords) - min(x_coords)
        box_h = max(y_coords) - min(y_coords)
        
        return box_cx, box_cy, box_w, box_h

    def compute_rotation(self, landmarks):
        import math
        dx = landmarks[12].x - landmarks[0].x
        dy = landmarks[12].y - landmarks[0].y
        rotation = math.atan2(dy, dx) * (180 / math.pi)
        return rotation

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        text_offset_y = 50

        if results.multi_hand_landmarks:
            for index, (hand_landmarks, handness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks.landmark)
                hand_type = "Left" if handness.classification[0].label == "Left" else "Right"
                if gesture:
                    cv2.putText(frame, f"{hand_type} hand {gesture}", (50, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    text_offset_y += 40

                box_cx, box_cy, box_w, box_h = self.compute_bounding_box(hand_landmarks.landmark)
                rotation = self.compute_rotation(hand_landmarks.landmark)
                label = self.detect_gesture(hand_landmarks.landmark)
                Number = 1 if handness.classification[0].label == "Left" else 2

                if self.json_output:
                    output = {
                        "Number": Number,
                        "box_cx": box_cx,
                        "box_cy": box_cy,
                        "box_w": box_w,
                        "box_h": box_h,
                        "label": label,
                        "score": 1,
                        "rotation": round(rotation, 2)
                    }
                    with open(f"{self.output_path}_hand{index}.json", 'w') as f:
                        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture Recognition")
    parser.add_argument('-i', '--input', required=True, help="Path to input image or video")
    parser.add_argument('-o', '--output', required=True, help="Path to output image or JSON file")
    parser.add_argument('-n', '--no_image', action='store_true', help="Skip removing the image")
    parser.add_argument('-j', '--json', action='store_true', help="Output as JSON file")
    parser.add_argument('-p', '--play', action='store_true', help="Display the image or video")
    parser.add_argument('-t', '--threshold', type=float, default=0.06, help="Threshold for index finger direction detection")
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    gr = GestureRecognition(args.input, args.output, args.play, args.no_image, args.json, args.threshold)
    gr.run()
    