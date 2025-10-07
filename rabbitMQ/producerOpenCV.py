#!/usr/bin/env python
import pika
import time
import cv2
import numpy as np


# drone variabelen
LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])

# === CONFIG ===
POWER = 30        # snelheid (0–100)
DUR = 1           # duur per beweging (seconden)
libraryHD = 21
MIN_CONTOUR_AREA = 200
SHOW_DEBUG = True
TOLERANCE = 0.05   # tolerantie in meters voor positie correctie


# === Helperfuncties ===
def detect_red_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

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


# RabbitMQ verbinding
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')

# Webcam openen
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Kan de webcam niet openen")
    connection.close()
    exit()

counter = 1
last_send_time = time.time()

while True:
    # Lees webcamframe
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break
    
    frame_for_detection = frame.copy()
    red_objs, red_mask = detect_red_objects(frame_for_detection)

    # Toon het frame
    cv2.imshow("colour_mask", red_mask)
    cx, cy, area = max(red_objs, key=lambda x: x[2])

    print("position drone: x= " + cx, + "y= " + cy)
    #cv2.imshow("colour_mask", frame)

    # Verstuur elk 1 seconde een RabbitMQ-bericht
    if time.time() - last_send_time >= 1:
        message = f"Hello World! {counter}"
        channel.basic_publish(exchange='',
                              routing_key='hello',
                              body=message)
        print(f" [x] Sent '{message}'")
        counter += 1
        last_send_time = time.time()

    # Stop als je op 'q' drukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Alles netjes afsluiten
cap.release()
cv2.destroyAllWindows()
connection.close()
print("Programma gestopt.")
