import easyocr
import cv2
import math
import argparse
import os
import json

def compute_font_scale(text, width, height, font=cv2.FONT_HERSHEY_SIMPLEX, initial_scale=0.5):
    """Computes the optimal font scale to make the text fit within the specified width and height."""
    scale = initial_scale
    (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale=scale, thickness=2)
    
    while text_width < width * 0.8 and text_height < height * 0.8:
        scale += 0.1
        (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale=scale, thickness=2)
        
    while (text_width > width or text_height > height) and scale > 0.1:
        scale -= 0.1
        (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale=scale, thickness=2)
    
    return scale

def main(args):
    # Create an OCR reader instance for English
    reader = easyocr.Reader(['en'])

    # Read from an image file
    img = cv2.imread(args.input)
    result = reader.readtext(args.input)
    Count_Detect = 0
    outputs = []

    # Process and display the results
    for detection in result:
        top_left = tuple(map(int, detection[0][0]))
        top_right = tuple(map(int, detection[0][1]))
        bottom_right = tuple(map(int, detection[0][2]))
        bottom_left = tuple(map(int, detection[0][3]))

        # Calculate center, width, and height of the bounding box
        box_cx = int((top_left[0] + bottom_right[0]) / 2)
        box_cy = int((top_left[1] + bottom_right[1]) / 2)
        box_w = int(bottom_right[0] - top_left[0])
        box_h = int(bottom_right[1] - top_left[1])

        # Calculate rotation angle in degrees
        delta_x = top_right[0] - top_left[0]
        delta_y = top_right[1] - top_left[1]
        rotation = math.degrees(math.atan2(delta_y, delta_x))

        text = detection[1]
        score = detection[2]  # Confidence score
        Count_Detect += 1

        # Output the bounding box properties in the specified format
        output = {
            "Number": Count_Detect,
            "box_cx": box_cx,
            "box_cy": box_cy,
            "box_w": box_w,
            "box_h": box_h,
            "label": text,
            "score": round(score, 3),
            "rotation": round(rotation, 2)
        }
        outputs.append(output)

        # Compute appropriate font scale for the bounding box
        font_scale = compute_font_scale(text, box_w, box_h)
    
        # Draw rectangle around detected text
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    
        # Display text inside rectangle
        bottom_left_text = (top_left[0], top_left[1] + int(box_h * 0.9))
        img = cv2.putText(img, text, bottom_left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

    # Save the image if output path is provided
    if args.output:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(args.output, img)

    # Save the results as JSON if the json flag is provided
    if args.json:
        json_path = os.path.splitext(args.output)[0] + '.json' if args.output else 'output.json'
        with open(json_path, 'w') as json_file:
            json.dump(outputs, json_file, indent=4)

    # Display the image unless no_image flag is provided
    if not args.no_image:
        cv2.imshow('Annotated Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Image Processing")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", help="Path to save the output image")
    parser.add_argument("-n", "--no_image", action="store_true", help="Skip displaying the image")
    parser.add_argument("-j", "--json", action="store_true", help="Output the results as a JSON file")
    args = parser.parse_args()
    main(args)
