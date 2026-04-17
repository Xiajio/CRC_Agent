import os
import shutil
from ultralytics import YOLO
import cv2

def screen_images(
    input_dir: str,
    output_dir: str,
    model_path: str,
    confidence_threshold: float = 0.5
):
    """
    Screens images using a trained YOLOv5 model and copies images with detected tumors
    to the output directory while maintaining the folder structure.

    Args:
        input_dir (str): Path to the input images directory.
        output_dir (str): Path to the output images directory.
        model_path (str): Path to the trained YOLOv5 model file.
        confidence_threshold (float, optional): Confidence threshold for detections.
                                               Defaults to 0.5.
    """
    # Load the trained YOLOv5 model
    try:
        print(f"Loading YOLOv11 model from {model_path}...")
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize counters
    total_images = 0
    images_with_tumor = 0
    images_without_tumor = 0

    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                total_images += 1
                input_image_path = os.path.join(root, file)

                # Compute the relative path to maintain folder structure
                relative_path = os.path.relpath(root, input_dir)
                output_image_dir = os.path.join(output_dir, relative_path)
                output_image_path = os.path.join(output_image_dir, file)

                # Ensure the output directory exists
                os.makedirs(output_image_dir, exist_ok=True)

                # Read the image using OpenCV
                image = cv2.imread(input_image_path)
                if image is None:
                    print(f"Warning: Unable to read image at {input_image_path}. Skipping.")
                    images_without_tumor += 1
                    continue

                # Run the model on the image
                try:
                    results = model.predict(source=image, verbose=False)
                except Exception as e:
                    print(f"Error processing image {input_image_path}: {e}")
                    images_without_tumor += 1
                    continue

                # Check for detections above the confidence threshold
                has_tumor = False
                for result in results:
                    for box in result.boxes:
                        confidence = box.conf.item()  # Confidence score
                        if confidence >= confidence_threshold:
                            has_tumor = True
                            break  # No need to check further if one tumor is found
                    if has_tumor:
                        break

                if has_tumor:
                    # Copy the image to the output directory
                    try:
                        shutil.copy2(input_image_path, output_image_path)
                        images_with_tumor += 1
                        print(f"Copied: {input_image_path} -> {output_image_path}")
                    except Exception as e:
                        print(f"Error copying image {input_image_path} to {output_image_path}: {e}")
                else:
                    images_without_tumor += 1
                    print(f"Filtered out: {input_image_path}")

    # Summary of the screening process
    print("\n--- Screening Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Images with tumors (copied): {images_with_tumor}")
    print(f"Images without tumors (filtered out): {images_without_tumor}")
    print(f"Filtered images are saved in: {output_dir}")

def main():
    # Configuration
    input_directory = r'J:\NAC_POST_PNG'
    output_directory = r'J:\NAC_POST_PNG_Filter'
    model_file_path = r'E:\LangG\src\tools\tool\Tumor_Detection\best.pt'  # Update this path if different
    confidence_thresh = 0.60  # Adjust based on desired sensitivity

    # Ensure input directory exists
    if not os.path.exists(input_directory):
        print(f"Input directory does not exist: {input_directory}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Start screening images
    screen_images(
        input_dir=input_directory,
        output_dir=output_directory,
        model_path=model_file_path,
        confidence_threshold=confidence_thresh
    )

if __name__ == '__main__':
    main()
