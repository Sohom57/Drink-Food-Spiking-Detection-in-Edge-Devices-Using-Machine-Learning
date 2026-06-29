import cv2
import time
import mediapipe as mp

from utils_download import download_models
from module_hands import HandTracker
from module_faces import FaceTracker
from module_objects import FoodDetector
from module_metrics import PerformanceEvaluator

def main():
    download_models()

    print("Initializing AI Classes...")
    metrics_tracker = PerformanceEvaluator() 
    hand_tracker = HandTracker()
    face_tracker = FaceTracker("user.jpg")
    food_tracker = FoodDetector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    MIRROR_WEBCAM = False 
    last_timestamp_ms = 0
    frame_count = 0
    print("Pipeline running! Press 'q' to quit.")

    while cap.isOpened():
        loop_start_time = time.time()
        
        success, img = cap.read()
        if not success:
            continue

        if MIRROR_WEBCAM:
            img = cv2.flip(img, 1)

        clean_img = img.copy() 
        img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        # --- 1. Run Modules ---
        hands_data = hand_tracker.process_and_return(mp_image, timestamp_ms)
        detected_food = food_tracker.process_and_draw(img, clean_img) 
        if detected_food is None: detected_food = []
        
        face_tracker.process_and_draw(img, clean_img, mp_image, timestamp_ms, frame_count)

        # --- 2. Stage C & D: Spatial ROI Masking & Biometric Intent Check ---
        user_is_authenticated = "User Present" in face_tracker.current_label

        for food in detected_food:
            x, y, w, h = food["box"]
            category = food["category"]
            
            rx1, ry1 = x, y
            rx2, ry2 = x + w, y + h
            
            tampering_detected = False

            # Check if any part of the hand intersects with the object's bounding box
            for hand_lms in hands_data:
                for pt in hand_lms:
                    pt_x, pt_y = pt
                    if rx1 < pt_x < rx2 and ry1 < pt_y < ry2:
                        tampering_detected = True
                        break
                if tampering_detected:
                    break
            
            if tampering_detected and not user_is_authenticated:
                # Tampered State (Red)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 4)
                cv2.putText(img, f"WARNING: {category.upper()} TAMPERING!", (rx1, ry1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Secure State (Blue)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 100, 0), 2)
                cv2.putText(img, f"{category.upper()} (Secure)", (rx1, ry1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)

        # Draw the hand skeleton matching the video style
        for hand_lms in hands_data:
            # Draw bones (green lines)
            for connection in hand_tracker.connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_lms) and end_idx < len(hand_lms):
                    cv2.line(img, hand_lms[start_idx], hand_lms[end_idx], (0, 255, 0), 2)
            
            # Draw yellow dots at fingertips (4, 8, 12, 16, 20) and wrist (0)
            key_points = [0, 4, 8, 12, 16, 20]
            for idx in key_points:
                if idx < len(hand_lms):
                    cv2.circle(img, hand_lms[idx], 6, (0, 255, 255), -1)

        # --- 3. Performance Metrics HUD ---
        metrics_tracker.update_and_draw(img, loop_start_time)

        # Changed window title to match the video
        cv2.imshow("Owner-Aware Security Pipeline", img)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hand_tracker.close()
    face_tracker.close()
    food_tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()