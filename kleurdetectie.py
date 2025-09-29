import cv2
import numpy as np

# ---------- Configuratie ----------
VIDEO_SOURCE = 0
SHOW_DEBUG = True
MIN_CONTOUR_AREA = 200  # Kleinere waarde voor test

# HSV ranges
# Rood (wrap-around)
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# Groen
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Geel
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

# ---------- Helper functies ----------
def detect_red_objects(frame):
    """Detecteert rode objecten en retourneert lijst van (cx, cy, area)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        objects.append((cx, cy, area))
    return objects, mask, contours

def detect_colored_objects(frame, lower, upper):
    """Detecteert gekleurde objecten zoals groen of geel"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        objects.append((cx, cy, area))
    return objects, mask, contours

def apply_homography(H, pt):
    """pt = (x, y) pixel -> return (X, Y) in wereld-eenheden (meters)"""
    src = np.array([[pt[0], pt[1], 1]]).T
    dst = H.dot(src)
    dst /= dst[2,0]
    return float(dst[0,0]), float(dst[1,0])

# ---------- Main loop ----------
def main():
    global H
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Kon camera niet openen")
        return

    H = None  # Homography matrix, optioneel
    print("Start tracking. Druk 'q' om te stoppen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        drone_world_coords = {}

        # ---- Rood ----
        red_objs, red_mask, _ = detect_red_objects(frame)
        for cx, cy, area in red_objs:
            drone_world_coords.setdefault("red", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
            cv2.putText(frame, f"red ({cx},{cy})", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # ---- Groen (was blauw) ----
        green_objs, green_mask, _ = detect_colored_objects(frame, LOWER_GREEN, UPPER_GREEN)
        for cx, cy, area in green_objs:
            drone_world_coords.setdefault("green", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
            cv2.putText(frame, f"green ({cx},{cy})", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # ---- Geel ----
        yellow_objs, yellow_mask, _ = detect_colored_objects(frame, LOWER_YELLOW, UPPER_YELLOW)
        for cx, cy, area in yellow_objs:
            drone_world_coords.setdefault("yellow", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0,255,255), -1)
            cv2.putText(frame, f"yellow ({cx},{cy})", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # ---- Debug vensters ----
        if SHOW_DEBUG:
            cv2.imshow("Frame", frame)
            cv2.imshow("Red Mask", red_mask)
            cv2.imshow("Green Mask", green_mask)
            cv2.imshow("Yellow Mask", yellow_mask)

        if drone_world_coords:
            print("Drone posities:", drone_world_coords)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
