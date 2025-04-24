# utils.py - Helper functions for insect detection demo

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

def perform_detection(
    yolo_model: YOLO,
    frame: np.ndarray,
    conf_threshold: float=0.5
) -> Optional[List[Dict]]:
    """
    Runs the YOLO model inference on a single frame.
    """
    if frame is None:
        print("Error: Input frame is None in perform_detection.")
        return None
    try:
        # Perform inference using the model
        results = yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
        return results
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

def create_motion_mask(frame, threshold=25):
    """
    Creates a simple motion mask from an image.
    For the demo, we'll use a basic thresholding approach.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, threshold
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

def postprocess_results(
    results: Optional[List[Dict]],
    model_class_names: Dict[int, str],
    mask: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Extracts information from detection results.
    If a mask is provided, only keeps detections that overlap with the mask.
    """
    detections_list = []
    if results is None or not results:
        return detections_list

    try:
        boxes = results[0].boxes
    except (IndexError, AttributeError) as e:
        print(f"Warning: Could not access boxes in results: {e}")
        return detections_list

    for box in boxes:
        try:
            # Extract bounding box coordinates (xyxy format)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # If we have a mask, check if this detection overlaps with it
            if mask is not None:
                # Check if the center of the bounding box is within the mask
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Also check if a significant portion of the box overlaps with the mask
                # First make sure we stay within mask boundaries
                y1_safe = max(0, min(y1, mask.shape[0]-1))
                y2_safe = max(0, min(y2, mask.shape[0]-1))
                x1_safe = max(0, min(x1, mask.shape[1]-1))
                x2_safe = max(0, min(x2, mask.shape[1]-1))
                
                # Extract the region of the mask corresponding to the bounding box
                box_region = mask[y1_safe:y2_safe, x1_safe:x2_safe]
                
                # Calculate overlap
                if box_region.size > 0:
                    mask_coverage = np.sum(box_region > 0) / box_region.size
                else:
                    mask_coverage = 0
                
                # Skip this detection if it doesn't overlap with the mask
                if not (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and 
                       (mask[center_y, center_x] > 0 or mask_coverage > 0.5)):
                    continue

            # Extract confidence score
            conf = float(box.conf[0].cpu().numpy())

            # Extract class ID and map to class name
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model_class_names.get(cls_id, f"Unknown Class {cls_id}")

            # Store detection info
            detections_list.append({
                'class_name': class_name,
                'confidence': conf,
                'bbox_xyxy': [x1, y1, x2, y2]
            })
        except Exception as e:
            print(f"Error processing a detection box: {e}")
            continue
    return detections_list

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws bounding boxes and labels on the frame.
    If mask is provided, overlays it on the frame.
    """
    output_frame = frame.copy()
    
    # If we have a mask, overlay it with transparency
    if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
        # Create a colored mask for overlay
        mask_overlay = np.zeros_like(output_frame)
        mask_overlay[mask > 0] = [0, 100, 0]  # Green tint for mask regions
        
        # Blend mask with the frame
        output_frame = cv2.addWeighted(output_frame, 0.7, mask_overlay, 0.3, 0)
    
    # Draw bounding boxes
    color = (0, 255, 0)  # Green color for bounding box
    font_scale = 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    for detection in detections:
        try:
            x1, y1, x2, y2 = detection['bbox_xyxy']
            class_name = detection['class_name']
            conf = detection['confidence']

            # Draw Bounding Box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 5)

            # Prepare and Draw Label
            label = f"{class_name}: {conf:.2f}"

            # Calculate text size for background
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, 3)
            label_ymin = max(y1, label_height + 10)

            # Draw background for text
            cv2.rectangle(output_frame,
                          (x1, label_ymin - label_height - 10),
                          (x1 + label_width, label_ymin - baseline),
                          color,
                          cv2.FILLED)

            # Add text
            cv2.putText(output_frame,
                        label,
                        (x1, label_ymin - 5),
                        font,
                        font_scale,
                        (255, 255, 255),  # White color
                        3)
        except Exception as e:
            continue
    return output_frame

def load_yolo_model(model_path):
    """
    Loads the YOLO model from the specified path.
    """
    print("Loading the YOLO model...")
    try:
        model = YOLO(model_path)
        class_names = model.names
        print(f"Model loaded with {len(class_names)} classes!")
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_image(image_path):
    """
    Loads an image from the specified path.
    """
    print(f"Opening image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file '{image_path}'.")
    return image

def load_or_create_mask(image, mask_path=None):
    """
    Either loads a mask from disk or creates a new one from the image.
    """
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask: {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        print("Creating mask from image...")
        mask = create_motion_mask(image)
    
    return mask

def display_results(output_frame, detections, mask=None):
    """
    Displays detection results and saves the output image.
    """
    # Display results in console
    print("\n--- Insects Detected ---")
    if detections:
        for i, obj in enumerate(detections, 1):
            print(f"{i}. {obj['class_name']} (confidence: {obj['confidence']:.2f})")
    else:
        print("No insects detected.")
    
    # Save mask if it exists
    if mask is not None:
        cv2.imwrite("motion_mask.png", mask)
        print("Motion mask saved to: motion_mask.png")
    
    # Convert from BGR to RGB for display
    output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(output_rgb)
    plt.title("Insect Detection Results")
    plt.axis('off')
    plt.show()
    
    # Save result
    result_path = "detection_result.jpg"
    cv2.imwrite(result_path, output_frame)
    print(f"Result saved to: {result_path}")

import os  # Added for file path operations