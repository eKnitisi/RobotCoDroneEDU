# move_one_meter_x.py
import cv2
import numpy as np
import stag
import time

# --- Configuratie ---
VIDEO_SOURCE = 0
SHOW_DEBUG = True
MIN_CONTOUR_AREA = 200

# STag bibliotheek (marker set)
libraryHD = 21

# Wereldcoördinaten van de STag-markers (meters)
world_points = {
    0: (0.0, 0.0),
    1: (2.04, 0.0),
    2: (2.04, 2.22),
    3: (0.0, 2.22)
}

# HSV kleurranges
LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])

# --- Helper functies ---

def detect_red_objects(frame):
    """Detecteer rode blobs en retourneer lijst van (cx, cy, area), plus mask."""
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

def apply_homography(H, pt):
    """pt = (u, v) pixel -> return (X, Y) in wereld-eenheden (meters)."""
    src = np.array([[pt[0], pt[1], 1.0]]).T
    dst = H.dot(src)
    dst /= dst[2, 0]
    return float(dst[0, 0]), float(dst[1, 0])

# Probeer robotlink te importeren; anders simuleren we
try:
    import robotlink as rl
    ROBOTLINK_AVAILABLE = True
except Exception:
    ROBOTLINK_AVAILABLE = False

def send_move_command(drone_id, x, y, z):
    """Verstuur move-commando naar drone of simuleer."""
    if ROBOTLINK_AVAILABLE:
        # Pas aan naar echte API van robotlink als nodig
        print(f"[RobotLink] moving drone {drone_id} -> ({x:.2f}, {y:.2f}, {z:.2f})")
        rl.move_to(drone_id=drone_id, x=float(x), y=float(y), z=float(z))
    else:
        print(f"[SIM] move drone {drone_id} -> ({x:.2f}, {y:.2f}, {z:.2f})")

CLOSED_LOOP_STEP = 0.3  # afstand per commando in meters (~tijd per stap)
POWER = 30
DUR = 0.5               # korte beweging voor nauwkeurigheid

# --- Main ---
def main():
    global kill
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kan camera niet openen.")
        return

    H = None
    print("Start. Druk 'c' voor calibratie, 'm' voor autonoom +1m X, 'q' voor noodstop.")

    while True:
        ret, frame = cap.read()
        if not ret or kill:
            break

        # Detecteer markers en rode drone
        corners, ids, _ = stag.detectMarkers(frame, libraryHD)
        if ids is not None:
            stag.drawDetectedMarkers(frame, corners, ids)

        red_objs, red_mask = detect_red_objects(frame)
        drone_world_coords = {}

        if H is not None and red_objs:
            cx, cy, area = max(red_objs, key=lambda x: x[2])
            X_curr, Y_curr = apply_homography(H, (cx, cy))
            drone_world_coords["red"] = (X_curr, Y_curr)
            cv2.putText(frame, f"red: ({X_curr:.2f},{Y_curr:.2f}) m", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

        if SHOW_DEBUG:
            cv2.imshow("Frame", frame)
            if 'red_mask' in locals():
                cv2.imshow("Red Mask", red_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or kill:
            break

        # Calibratie
        if key == ord('c') and ids is not None and len(ids) >= 4:
            image_pts, world_pts = [], []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in world_points:
                    c = corners[i][0]
                    center = np.mean(c, axis=0)
                    image_pts.append(center)
                    world_pts.append(world_points[marker_id])
            if len(image_pts) >= 4:
                H, _ = cv2.findHomography(np.array(image_pts), np.array(world_pts))
                np.save("homography.npy", H)
                print("✅ Homografie berekend en opgeslagen.")

        # --- Autonome beweging +1m X (closed-loop) ---
        if key == ord('m') and not kill:
            if H is None:
                print("⚠️ Eerst 'c' drukken voor calibratie!")
                continue
            if "red" not in drone_world_coords:
                print("⚠️ Rode drone niet gedetecteerd!")
                continue

            print("\n=== Closed-loop vlucht gestart ===")
            swarm.takeoff()
            swarm.hover(1)

            X_target = drone_world_coords["red"][0] + 1.0  # +1 meter
            Y_target = drone_world_coords["red"][1]

            # Blijf bewegen totdat we bij target zijn
            while not kill:
                ret, frame = cap.read()
                if not ret:
                    break
                red_objs, _ = detect_red_objects(frame)
                if not red_objs:
                    print("Drone niet zichtbaar, wacht...")
                    time.sleep(0.2)
                    continue
                cx, cy, area = max(red_objs, key=lambda x: x[2])
                X_curr, Y_curr = apply_homography(H, (cx, cy))

                error_X = X_target - X_curr
                if abs(error_X) < 0.05:  # tolerantie 5 cm
                    print(f"✅ Doelpositie X bereikt: {X_curr:.2f} m")
                    break

                # Kies richting
                if error_X > 0:
                    swarm.go("right", POWER, DUR)
                else:
                    swarm.go("left", POWER, DUR)

                swarm.hover(0.3)

            swarm.land()
            swarm.disconnect()
            print("✅ Vlucht voltooid.")
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()
if __name__ == "__main__":
    main()
