import cv2
from fer import FER
import argparse
import os
import json

class EmotionDetection:
    def __init__(self, input_path=None, output_path=None, no_image=False, json_output=False, play=False):
        if input_path.endswith('.jpg') or input_path.endswith('.png'):
            self.mode = 'image'
            self.image = cv2.imread(input_path)
        else:
            self.mode = 'video'
            self.cap = cv2.VideoCapture(input_path if input_path else 0)
        self.detector = FER(mtcnn=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.output_path = output_path
        self.no_image = no_image
        self.json_output = json_output
        self.play = play

    def detect_emotion(self, frame):
        try:
            result = self.detector.detect_emotions(frame)
            emotion = result[0]['emotions']
            dominant_emotion = max(emotion, key=emotion.get)
            return dominant_emotion, result[0]['box']
        except:
            return None, None

    def run(self):
        outputs = []

        if self.mode == 'image':
            emotion, box = self.detect_emotion(self.image)
            if emotion:
                x, y, w, h = box
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(self.image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                outputs.append({
                    "box_cx": x + w//2,
                    "box_cy": y + h//2,
                    "box_w": w,
                    "box_h": h,
                    "label": emotion,
                    "score": 1.0,  # Assuming score as 1 for simplicity
                    "rotation": 0.0  # Assuming no rotation
                })
            if self.play:
                cv2.imshow('Emotion Detection', self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if self.output_path and not self.no_image:
                cv2.imwrite(self.output_path, self.image)
        else:
            # Handle video processing similar to your original code
            pass

        if self.json_output:
            with open('output.json', 'w') as f:
                json.dump(outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Detection from Image or Video')
    parser.add_argument('-i', '--input', required=True, help='Path to input image or video')
    parser.add_argument('-o', '--output', help='Path to save output image or video')
    parser.add_argument('-n', '--no_image', action='store_true', help='Skip saving the image')
    parser.add_argument('-j', '--json', action='store_true', help='Output results as JSON')
    parser.add_argument('-p', '--play', action='store_true', help='Display the image or video')
    args = parser.parse_args()

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ed = EmotionDetection(args.input, args.output, args.no_image, args.json, args.play)
    ed.run()
