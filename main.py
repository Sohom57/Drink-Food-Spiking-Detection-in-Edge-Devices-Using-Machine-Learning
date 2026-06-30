import asyncio
import os
import time
from collections import deque

import cv2
import mediapipe as mp
from telegram import Bot

from utils_download import download_models
from module_hands import HandTracker
from module_faces import FaceTracker
from module_objects import FoodDetector
from module_metrics import PerformanceEvaluator

# --- Telegram Configuration ---
TELEGRAM_TOKEN = "8645978324:AAEemOi6rhyqggyh08qgMgVYrApTt-X89ng"
TELEGRAM_CHAT_ID = "5736374971"


def send_telegram_alert_sync(image_path, video_path):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = Bot(token=TELEGRAM_TOKEN)

        loop.run_until_complete(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚨 ALERT: Unauthorized Tampering Detected!"))

        with open(image_path, "rb") as img:
            loop.run_until_complete(bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img))

        with open(video_path, "rb") as vid:
            loop.run_until_complete(bot.send_video(chat_id=TELEGRAM_CHAT_ID, video=vid, caption="Event Buffer (5s)"))

    except Exception as e:
        print(f"[-] Failed to send Telegram alert: {e}")


def main():
    download_models()

    print("Initializing AI Classes...")
    metrics_tracker = PerformanceEvaluator()
    hand_tracker = HandTracker()
    face_tracker = FaceTracker("user.jpg")
    food_tracker = FoodDetector()

    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 30
    video_buffer = deque(maxlen=15 * fps)

    MIRROR_WEBCAM = False
    last_timestamp_ms = 0
    frame_count = 0
    any_tampering_detected = False
    tampering_alert_sent = False

    print("Pipeline running! Press 'q' to quit.")

    while cap.isOpened():
        loop_start_time = time.time()

        success, img = cap.read()
        if not success:
            continue

        video_buffer.append(img.copy())

        if MIRROR_WEBCAM:
            img = cv2.flip(img, 1)

        clean_img = img.copy()
        img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        hands_data = hand_tracker.process_and_return(mp_image, timestamp_ms)
        detected_food = food_tracker.process_and_draw(img, clean_img)
        if detected_food is None:
            detected_food = []

        face_tracker.process_and_draw(img, clean_img, mp_image, timestamp_ms, frame_count)

        user_is_authenticated = "User Present" in face_tracker.current_label

        for food in detected_food:
            x, y, w, h = food["box"]
            category = food["category"]

            rx1, ry1 = x, y
            rx2, ry2 = x + w, y + h

            tampering_detected = False

            for hand_lms in hands_data:
                for pt in hand_lms:
                    pt_x, pt_y = pt
                    if rx1 < pt_x < rx2 and ry1 < pt_y < ry2:
                        tampering_detected = True
                        break
                if tampering_detected:
                    break

            if tampering_detected and not user_is_authenticated:
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 4)
                cv2.putText(
                    img,
                    f"WARNING: {category.upper()} TAMPERING!",
                    (rx1, ry1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                metrics_tracker.log_tamper_event(category, "YES")
                any_tampering_detected = True

                if not tampering_alert_sent:
                    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

                    snapshot_path = os.path.join(evidence_dir, f"tamper_snapshot_{timestamp_str}.png")
                    cv2.imwrite(snapshot_path, img)
                    print(f"[!] Tamper snapshot saved to {snapshot_path}")

                    video_output_path = os.path.join(evidence_dir, f"tamper_buffer_{timestamp_str}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (1280, 720))

                    for frame in video_buffer:
                        out_video.write(frame)
                    out_video.release()
                    print(f"[!] Tamper video buffer saved to {video_output_path}")

                    send_telegram_alert_sync(snapshot_path, video_output_path)
                    tampering_alert_sent = True

            else:
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 100, 0), 2)
                cv2.putText(
                    img,
                    f"{category.upper()} (Secure)",
                    (rx1, ry1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 100, 0),
                    2,
                    cv2.LINE_AA,
                )

        for hand_lms in hands_data:
            for connection in hand_tracker.connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_lms) and end_idx < len(hand_lms):
                    cv2.line(img, hand_lms[start_idx], hand_lms[end_idx], (0, 255, 0), 2)

            key_points = [0, 4, 8, 12, 16, 20]
            for idx in key_points:
                if idx < len(hand_lms):
                    cv2.circle(img, hand_lms[idx], 6, (0, 255, 255), -1)

        metrics_tracker.update_and_draw(img, loop_start_time)

        cv2.imshow("Owner-Aware Security Pipeline", img)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    metrics_tracker.finalize_test_log(any_tampering_detected)

    hand_tracker.close()
    face_tracker.close()
    food_tracker.close()
    metrics_tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
