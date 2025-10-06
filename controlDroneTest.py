import cv2
import numpy as np
import stag
import threading
import keyboard
import time
import sys
from codrone_edu.swarm import *

# === CONFIG ===
POWER = 30        # snelheid (0–100)
DUR = 1           # duur per beweging (seconden)
libraryHD = 21
MIN_CONTOUR_AREA = 200
SHOW_DEBUG = True
TOLERANCE = 0.05   # tolerantie in meters voor positie correctie

# Lijst met waypoints (in meters)
waypoints = [
    (1.0, 1.0)  # voorlopig één punt, later meer toevoegen
]

# === STag wereldcoördinaten (meters) ===
world_points = {
    0: (0.0, 0.0),
    1: (2.04, 0.0),
    2: (2.04, 2.22),
    3: (0.0, 2.22)
}

# === HSV drempels voor rode drone ===
LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])

# === NOODSTOP ===
kill = False

def watch_for_q():
    global kill
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            kill = True
            try:
                print("Drone gaat veilig landen...")
                swarm.land()  # zachte landing
            except Exception as e:
                print("⚠️ Fout tijdens noodstop-landing:", e)
            finally:
                swarm.disconnect()
            break
        time.sleep(0.1)


# === Drone-setup ===
swarm = Swarm()
swarm.connect()

watcher = threading.Thread(target=watch_for_q, daemon=True)
watcher.start()

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

def apply_homography(H, pt):
    src = np.array([[pt[0], pt[1], 1.0]]).T
    dst = H.dot(src)
    dst /= dst[2, 0]
    return float(dst[0, 0]), float(dst[1, 0])


# === MAIN LOOP ===
def main():
    global kill
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kan camera niet openen.")
        return

    H = None
    print("Start. Druk 'c' voor calibratie, 'm' om autonoom +1m X te bewegen, 'q' voor noodstop.")

    while True:
        ret, frame = cap.read()
        if not ret or kill:
            break

        # --- Detectie rode drone ---
        frame_for_detection = frame.copy()
        red_objs, red_mask = detect_red_objects(frame_for_detection)
        drone_world_coords = {}

        if H is not None and red_objs:
            cx, cy, area = max(red_objs, key=lambda x: x[2])
            X, Y = apply_homography(H, (cx, cy))
            drone_world_coords["red"] = (X, Y)
            cv2.circle(frame, (cx, cy), 8, (0,0,255), -1)
            cv2.putText(frame, f"red: ({X:.2f},{Y:.2f}) m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # --- Detecteer markers zonder ID's ---
        corners, ids, _ = stag.detectMarkers(frame, libraryHD)
        if ids is not None:
            stag.drawDetectedMarkers(frame, corners, ids)

        # --- Teken verticale lijnen als H bekend is ---
        if H is not None:
            step = 0.5
            min_x = min(pt[0] for pt in world_points.values())
            max_x = max(pt[0] for pt in world_points.values())
            max_y = max(pt[1] for pt in world_points.values())
            H_inv = np.linalg.inv(H)

            for x in np.arange(min_x, max_x + step, step):
                top = np.array([[x, 0, 1]]).T
                bottom = np.array([[x, max_y, 1]]).T
                px_top = H_inv.dot(top)
                px_top /= px_top[2, 0]
                px_bottom = H_inv.dot(bottom)
                px_bottom /= px_bottom[2, 0]
                cv2.line(frame,
                        (int(px_top[0, 0]), int(px_top[1, 0])),
                        (int(px_bottom[0, 0]), int(px_bottom[1, 0])),
                        (0, 255, 0), 1)

        # --- Debug weergave ---
        if SHOW_DEBUG:
            cv2.imshow("Frame", frame)
            if red_mask is not None:
                cv2.imshow("Red Mask", red_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or kill:
            break

        # --- Calibratie ---
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
                print("Homografie berekend en opgeslagen.")

        # --- Hover mode na 'm' ---
        if key == ord('m') and not kill:
            if H is None:
                print("⚠️ Eerst 'c' drukken voor calibratie!")
                continue

            print("\n=== Drone opstijgen en hoveren ===")
            swarm.takeoff()
            swarm.hover(0.5)  # opstijgen en stabiliseren

            # Blijf frames lezen en drone positie tonen
            while not kill:
                ret, frame = cap.read()
                if not ret:
                    continue

                red_objs, _ = detect_red_objects(frame)
                if red_objs:
                    cx, cy, area = max(red_objs, key=lambda x: x[2])
                    X_curr, Y_curr = apply_homography(H, (cx, cy))
                    
                    # Terminal-output
                    print(f"Huidige positie -> X: {X_curr:.2f} m, Y: {Y_curr:.2f} m", end='\r', flush=True)

                    # Overlay op debug scherm
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"X: {X_curr:.2f}, Y: {Y_curr:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                else:
                    print("⚠️ Drone niet zichtbaar...                                ", end='\r', flush=True)

                if SHOW_DEBUG:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        kill = True
                        break

    swarm.hover(0.5)  # Houd hover als exit


    cap.release()
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception):
        print("\n>>> KeyboardInterrupt: drone gaat veilig landen <<<")
        try:
            swarm.land()
        except Exception as e:
            print("⚠️ Fout tijdens zachte noodlanding:", e)
        finally:
            swarm.disconnect()
        sys.exit()
