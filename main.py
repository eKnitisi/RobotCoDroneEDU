import cv2
import numpy as np
import time

# ---------- Configuratie ----------
VIDEO_SOURCE = 0  # 0 = standaard camera; of pad/rtsp stream
SHOW_DEBUG = True

# HSV ranges voor rood (let op: rood zit rond 0° en 180° in HSV -> 2 ranges)
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# Minimale contour oppervlakte om ruis te negeren (px)
MIN_CONTOUR_AREA = 500

# Kalman filter instellen (2D positie + snelheid)
use_kalman = True

# ---------- Helper functies ----------
def create_kalman():
    # state = [x, y, vx, vy]  (4x1)
    # measurement = [x, y]   (2x1)
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def detect_red_centroid(frame):
    """Retourneert (cx, cy, area, mask, contours) of None als niets gevonden."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphologische bewerkingen voor ruisreductie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, contours

    # Kies grootste contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None, mask, contours

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask, contours
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy, area), mask, contours

# ---------- Homography (pixels -> wereld coördinaten in meters) ----------
# Als je geen homography wilt gebruiken, laat H = None en je krijgt pixel coords.
H = None  # initialiseren; zie hieronder hoe te vullen

def compute_homography_from_points(image_points, world_points):
    """image_points: Nx2 pixel coords; world_points: Nx2 coords (meters). N >= 4"""
    image_pts = np.array(image_points, dtype=np.float32)
    world_pts = np.array(world_points, dtype=np.float32)
    H, status = cv2.findHomography(image_pts, world_pts, method=0)
    return H

def apply_homography(H, pt):
    """pt = (x, y) pixel -> return (X, Y) in wereld-eenheden (meters)"""
    if H is None:
        raise ValueError("Homography H is None")
    src = np.array([ [pt[0], pt[1], 1.0] ]).T
    dst = H.dot(src)
    dst /= dst[2,0]
    return float(dst[0,0]), float(dst[1,0])

# ---------- Main loop ----------
def main():
    global H
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Kon camera niet openen:", VIDEO_SOURCE)
        return

    # Optioneel: als je camera lens distorsie wilt corrigeren, doe camera calibratie eerst
    # (zie opmerkingen onderaan). Hier gebruiken we geen distortion correctie.

    kf = create_kalman() if use_kalman else None
    last_time = time.time()

    print("Start tracking. Druk 'q' om te stoppen. Druk 'h' om homography in te stellen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame niet gelezen")
            break

        # Detecteer rood object
        detection, mask, contours = detect_red_centroid(frame)

        measured = None
        if detection is not None:
            cx, cy, area = detection
            measured = np.array([[np.float32(cx)], [np.float32(cy)]])
            # Kalman update/predict
            if kf is not None:
                kf.predict()
                kf.correct(measured)
                state = kf.statePost
                px, py = int(state[0,0]), int(state[1,0])
            else:
                px, py = cx, cy

            # Teken marker en centroid
            cv2.drawContours(frame, [max(contours, key=cv2.contourArea)], -1, (0,255,0), 2)
            cv2.circle(frame, (px, py), 6, (255,0,0), -1)
            cv2.putText(frame, f"pixel: ({px},{py}) area:{int(area)}", (px+10, py-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Als homography is ingesteld: omzetting naar meters
            if H is not None:
                try:
                    X, Y = apply_homography(H, (px, py))
                    cv2.putText(frame, f"world: ({X:.2f}m, {Y:.2f}m)", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                    # Print voor de drone controller (hier console)
                    print(f"time:{time.time():.3f}, px=({px},{py}), world=({X:.3f},{Y:.3f})")
                except Exception as e:
                    print("H fout:", e)
            else:
                print(f"time:{time.time():.3f}, px=({px},{py})")

        else:
            # geen detectie
            if use_kalman and kf is not None:
                # predict only
                pred = kf.predict()
                px, py = int(pred[0,0]), int(pred[1,0])
                # optioneel: teken voorspelling
                cv2.circle(frame, (px, py), 4, (0,255,255), -1)

        # Debug schermen
        if SHOW_DEBUG:
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            # TODO: gebruikersinstructie: pauzeer en klik 4 punten (image) en geef hun wereld coords
            # Voor eenvoudige workflow hieronder: capture current frame and ask user to input 4 img pts and world pts
            print("H: maak homography. Druk 's' om huidige frame te saven en handmatig 4 image points op te geven.")
            # sla huidige frame op
            cv2.imwrite("homography_sample.png", frame)
            print("Frame opgeslagen als homography_sample.png. Open het en noteer 4 pixel-coördinaten (x,y).")
            print("Geef die 4 punten in als: x1,y1 x2,y2 x3,y3 x4,y4 (volgorde moet corresponderen met wereld punten)")
            inp = input("image points: ")
            try:
                img_pts = []
                for pair in inp.split():
                    x,y = pair.split(',')
                    img_pts.append((float(x), float(y)))
                if len(img_pts) != 4:
                    print("Voer precies 4 punten in.")
                    continue
                print("Voer nu 4 wereldpunten (meters) in dezelfde volgorde, bvb: 0,0 2,0 2,3 0,3")
                inp2 = input("world points: ")
                world_pts = []
                for pair in inp2.split():
                    x,y = pair.split(',')
                    world_pts.append((float(x), float(y)))
                if len(world_pts) != 4:
                    print("Voer precies 4 wereldpunten in.")
                    continue
                H = compute_homography_from_points(img_pts, world_pts)
                print("Homography ingesteld.")
            except Exception as e:
                print("Fout bij instellen homography:", e)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
