import cv2
import numpy as np
import stag

# ===============================
# CONFIGURATIE
# ===============================

VIDEO_SOURCE = 0  # Camera ID
SHOW_DEBUG = True
MIN_CONTOUR_AREA = 200

# STag bibliotheek (marker set)
libraryHD = 21

# Wereldcoördinaten van de STag-markers (in meters)
world_points = {
    0: (0.0, 0.0),      # Marker ID 0
    1: (2.04, 0.0),     # Marker ID 1
    2: (2.04, 2.22),    # Marker ID 2
    3: (0.0, 2.22)      # Marker ID 3
}

# Kleurfilters (HSV)
LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

# ===============================
# HELPER FUNCTIES
# ===============================

def detect_colored_objects(frame, lower, upper):
    """Detecteert gekleurde objecten (zoals groen, geel of rood)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objects.append((cx, cy, area))
    return objects, mask, contours


def detect_red_objects(frame):
    """Detecteert rode objecten (combinatie van 2 HSV-bereiken)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objects.append((cx, cy, area))
    return objects, mask, contours


def apply_homography(H, pt):
    """Zet pixelpunt om naar wereldcoördinaten (meter)."""
    src = np.array([[pt[0], pt[1], 1]]).T
    dst = H.dot(src)
    dst /= dst[2, 0]
    return float(dst[0, 0]), float(dst[1, 0])


# ===============================
# MAIN PROGRAMMA
# ===============================

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kon camera niet openen.")
        return

    H = None
    print("✅ Camera gestart. Druk 'c' om homografie te berekenen, 'q' om te stoppen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ========== STAG MARKER DETECTIE ==========
        corners, ids, _ = stag.detectMarkers(frame, libraryHD)

        if ids is not None and len(ids) > 0:
            stag.drawDetectedMarkers(frame, corners, ids)

        # Als 4 markers gedetecteerd zijn, melding tonen
        if ids is not None and len(ids) >= 4 and H is None:
            cv2.putText(frame, "4 markers gedetecteerd. Druk 'c' om homografie te berekenen.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ========== DRONE DETECTIE ==========
        drone_world_coords = {}

        # Rood
        red_objs, red_mask, _ = detect_red_objects(frame)
        for cx, cy, area in red_objs:
            drone_world_coords.setdefault("red", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Groen
        green_objs, green_mask, _ = detect_colored_objects(frame, LOWER_GREEN, UPPER_GREEN)
        for cx, cy, area in green_objs:
            drone_world_coords.setdefault("green", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        # Geel
        yellow_objs, yellow_mask, _ = detect_colored_objects(frame, LOWER_YELLOW, UPPER_YELLOW)
        for cx, cy, area in yellow_objs:
            drone_world_coords.setdefault("yellow", []).append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        # ========== PIXEL → WERELDCOÖRDINATEN ==========
        if H is not None:
            for color, positions in drone_world_coords.items():
                world_positions = []
                for (cx, cy) in positions:
                    X, Y = apply_homography(H, (cx, cy))
                    world_positions.append((X, Y))
                drone_world_coords[color] = world_positions

        # ========== TEKST IN BEELD ==========
        for color, positions in drone_world_coords.items():
            for (X, Y) in positions:
                cv2.putText(frame, f"{color}: ({X:.2f}, {Y:.2f}) m",
                            (20, 40 if color == 'red' else 60 if color == 'green' else 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ========== DEBUG WINDOWS ==========
        if SHOW_DEBUG:
            cv2.imshow("Frame", frame)
            if H is None:  # alleen tonen bij calibratie
                cv2.imshow("Red Mask", red_mask)
                cv2.imshow("Green Mask", green_mask)
                cv2.imshow("Yellow Mask", yellow_mask)

        # Print posities in terminal
        if drone_world_coords:
            print("Drone posities (wereld):", drone_world_coords)

        # ========== TOETSEN ==========
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and ids is not None and len(ids) >= 4:
            # Bereken homografie
            image_pts = []
            world_pts = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in world_points:
                    c = corners[i][0]
                    center = np.mean(c, axis=0)
                    image_pts.append(center)
                    world_pts.append(world_points[marker_id])

            if len(image_pts) == 4:
                H, _ = cv2.findHomography(np.array(image_pts), np.array(world_pts))
                np.save("homography.npy", H)
                print("✅ Homografie berekend en opgeslagen als homography.npy")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
