from ultralytics import YOLO
import cv2
import numpy as np
import base64
import time

def write_image_cv2(image_path, image):
    # Write the image to file
    cv2.imwrite(image_path, image)

def count_damage(results):
    total_area = 0
    area_list = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()   # Retrieve masks as numpy arrays
        image_area = masks.shape[1] * masks.shape[2]  # Calculate total number of pixels in the image
        for i, mask in enumerate(masks):
            binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary
            color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert binary mask to color
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
            contour = contours[0]  # Retrieve the first contour
            area = cv2.contourArea(contour)  # Calculate the area of the pothole
            area_list.append(area)  # Append area to the list
            cv2.drawContours(color_mask, [contour], -1, (0, 255, 0), 3)  # Draw the contour on the mask
        
        for i, area in enumerate(area_list):

            total_area += area  # Sum the areas for total

        return (total_area / image_area) * 100
    else:
        return 0
    

def predict(filename):
    model = YOLO('./model/best.pt')
    results = model.predict(f'''./tmp/{filename}''', imgsz=640)
    # Load the image from file
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    unix_time = int(time.time())
    
    damage = count_damage(results)

    if(damage > 0):
        write_image_cv2(f'''./result/{unix_time}.jpg''', annotated_image_rgb)

        response = {
            "filename": f'''{unix_time}.jpg''',
            "is_damaged": True,
            "damage_precent": damage
        }

        return response

    return {
        "is_damaged": False
    }