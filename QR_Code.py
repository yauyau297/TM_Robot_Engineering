import cv2
import argparse
import os
import json
from pyzbar.pyzbar import decode

def detect_and_decode_codes(input_path, output_path=None, no_image=False, json_output=False, play=False):
    if input_path.endswith('.mp4') or input_path.endswith('.avi'):
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, output_path, no_image, json_output, play)
            if play:
                cv2.imshow('QR and Barcode Decoder', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
    else:
        image = cv2.imread(input_path)
        process_frame(image, output_path, no_image, json_output, play)
        if play:
            cv2.imshow('QR and Barcode Decoder', image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_frame(image, output_path, no_image, json_output, play):
    codes = decode(image)
    Count_Detect = 0
    outputs = []
    for code in codes:
        if len(code.polygon) == 4:
            pts = [tuple(pt) for pt in code.polygon]
            for i in range(4):
                cv2.line(image, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
        else:
            x, y, w, h = code.rect
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = code.data.decode('utf-8')
        cv2.putText(image, text, (code.rect[0], code.rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        Count_Detect += 1
        
        # Assuming some default values for score and rotation as they are not provided in the original code
        output = {
            "Number": Count_Detect,
            "box_cx": code.rect[0] + code.rect[2] // 2,
            "box_cy": code.rect[1] + code.rect[3] // 2,
            "box_w": code.rect[2],
            "box_h": code.rect[3],
            "label": text,
            "score": 0.99,  # Default value
            "rotation": 0.0  # Default value
        }
        outputs.append(output)

    if json_output:
        with open(output_path + '.json', 'w') as f:
            json.dump(outputs, f)

    if not no_image:
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QR and Barcode Decoder')
    parser.add_argument('-i', '--input', required=True, help='Path to the input image or video')
    parser.add_argument('-o', '--output', required=True, help='Path to the output image or directory')
    parser.add_argument('-n', '--no_image', action='store_true', help='Skip saving the image')
    parser.add_argument('-j', '--json', action='store_true', help='Output the results as a JSON file')
    parser.add_argument('-p', '--play', action='store_true', help='Display the image or video')

    args = parser.parse_args()

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detect_and_decode_codes(args.input, args.output, args.no_image, args.json, args.play)
