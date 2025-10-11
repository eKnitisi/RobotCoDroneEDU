#!/usr/bin/env python
import pika
import time
import cv2
import numpy as np
import json
import stag

# === CONFIG ===
FPS_POSITION_UPDATE = 15
MIN_CONTOUR_AREA = 200
libraryHD = 21

# === Drone kleurvariabelen ===
LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])

LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([80, 255, 255])

LOWER_YELLOW = np.array([20, 50, 50])
UPPER_YELLOW = np.array([30, 255, 255])

# === Wereldcoördinaten van STag markers (meters) ===
world_points = {
    0: (0.0, 0.0),
    1: (2.04, 0.0),
    2: (2.04, 2.22),
    3: (0.0, 2.22)
}

H = None  # homografie matrix, wordt later berekend
# Voor testen zonder echte markers: dummy homografie (identiteit)
H = np.eye(3)

def apply_homography(H, pt):
    """Pixel coördinaten naar wereldcoördinaten"""
    src = np.array([[pt[0], pt[1], 1.0]]).T
    dst = H.dot(src)
    dst /= dst[2,0]
    return float(dst[0,0]), float(dst[1,0])


# === Helperfuncties ===
def detect_color_objects(frame, lower_bounds, upper_bounds):
    """Algemene functie voor het detecteren van objecten in een kleur"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if isinstance(lower_bounds, list):
        mask = cv2.inRange(hsv, lower_bounds[0], upper_bounds[0])
        for i in range(1, len(lower_bounds)):
            mask_i = cv2.inRange(hsv, lower_bounds[i], upper_bounds[i])
            mask = cv2.bitwise_or(mask, mask_i)
    else:
        mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        objs.append((cx, cy, area))
    return objs, mask

# === RabbitMQ verbinding ===
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')

# === Webcam openen ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Kan de webcam niet openen")
    connection.close()
    exit()

counter = 1
last_update_time = 0

# === Hoofdlus ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    # --- Detecteer markers voor calibratie ---
    corners, ids, _ = stag.detectMarkers(frame, libraryHD)
    if ids is not None:
        stag.drawDetectedMarkers(frame, corners, ids)

    key = cv2.waitKey(1) & 0xFF

    # --- Calibratie op 'c' ---
    if key == ord('c') and ids is not None and len(ids) >= 4:
        image_pts, world_pts_list = [], []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in world_points:
                c = corners[i][0]
                center = np.mean(c, axis=0)
                image_pts.append(center)
                world_pts_list.append(world_points[marker_id])
        if len(image_pts) >= 4:
            H, _ = cv2.findHomography(np.array(image_pts), np.array(world_pts_list))
            print("✅ Homografie berekend en opgeslagen.")

    # --- Wacht tot calibratie is gedaan ---
    if H is None:
        cv2.putText(frame, "Druk 'c' om grid te calibreren", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("drone_position", frame)
        continue  # spring naar volgende iteratie zonder drone-detectie

    # --- Drone detectie en RabbitMQ --- 
    #
    frame_for_detection = frame.copy()
    red_objs, _ = detect_color_objects(frame_for_detection, [LOWER_RED_1, LOWER_RED_2], [UPPER_RED_1, UPPER_RED_2])
    green_objs, _ = detect_color_objects(frame_for_detection, LOWER_GREEN, UPPER_GREEN)
    yellow_objs, _ = detect_color_objects(frame_for_detection, LOWER_YELLOW, UPPER_YELLOW)
 
    if time.time() - last_update_time >= 1 / FPS_POSITION_UPDATE:

        color_data = []

        # RODE drone
        if red_objs:
            cx, cy, _ = max(red_objs, key=lambda x: x[2])
            X_w, Y_w = apply_homography(H, (cx, cy))
            color_data.append(("red", X_w, Y_w))
            cv2.circle(frame, (cx, cy), 10, (0,0,255), -1)
            cv2.putText(frame, f"({X_w:.2f},{Y_w:.2f})", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # GROENE drone
        if green_objs:
            cx, cy, _ = max(green_objs, key=lambda x: x[2])
            X_w, Y_w = apply_homography(H, (cx, cy))
            color_data.append(("green", X_w, Y_w))
            cv2.circle(frame, (cx, cy), 10, (0,255,0), -1)
            cv2.putText(frame, f"({X_w:.2f},{Y_w:.2f})", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # GELE drone
        if yellow_objs:
            cx, cy, _ = max(yellow_objs, key=lambda x: x[2])
            X_w, Y_w = apply_homography(H, (cx, cy))
            color_data.append(("yellow", X_w, Y_w))
            cv2.circle(frame, (cx, cy), 10, (0,255,255), -1)
            cv2.putText(frame, f"({X_w:.2f},{Y_w:.2f})", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Verstuur RabbitMQ-berichten
        for color, X_w, Y_w in color_data:
            message = {"id": counter, "color": color, "x": X_w, "y": Y_w, "timestamp": time.time()}
            channel.basic_publish(exchange='', routing_key='hello', body=json.dumps(message))
            print(f" [x] Sent {message}")
            counter += 1

        last_update_time = time.time()

    cv2.imshow("drone_position", frame)

    if key == ord('q'):
        break


# === Afsluiten ===
cap.release()
cv2.destroyAllWindows()
connection.close()
print("Programma gestopt.")
