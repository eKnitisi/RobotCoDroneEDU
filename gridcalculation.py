import stag
import cv2
import numpy as np

# Marker type instellen
libraryHD = 21

# Wereldcoördinaten van je grid (meters)
world_points = {
    0: (0, 0),   # Marker ID 00
    1: (2.04, 0),   # Marker ID 01
    2: (2.04, 2.22),   # Marker ID 02
    3: (0, 2.22)    # Marker ID 03
}

H = None  # Homography matrix

# Webcam openen
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera niet gevonden")
    exit()
else:
    print("Camera geopend")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    # Detecteer markers
    corners, ids, rejected_corners = stag.detectMarkers(frame, libraryHD)

    # Als 4 markers gedetecteerd zijn, toon een bericht
    if ids is not None and len(ids) >= 4 and H is None:
        cv2.putText(frame, "4 markers gedetecteerd. Druk 'c' om homografie te berekenen.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Als H al bekend is → grid tekenen
    if H is not None:
        step = 0.5
        H_inv = np.linalg.inv(H)

        # verticale lijnen
        for x in np.arange(0, 2.01, step):
            pt1 = np.array([[x, 0, 1]]).T
            pt2 = np.array([[x, 3, 1]]).T
            p1 = H_inv @ pt1
            p2 = H_inv @ pt2
            p1 /= p1[2, 0]
            p2 /= p2[2, 0]
            cv2.line(frame, (int(p1[0, 0]), int(p1[1, 0])),
                     (int(p2[0, 0]), int(p2[1, 0])), (0, 255, 0), 1)

        # horizontale lijnen
        for y in np.arange(0, 3.01, step):
            pt1 = np.array([[0, y, 1]]).T
            pt2 = np.array([[2, y, 1]]).T
            p1 = H_inv @ pt1
            p2 = H_inv @ pt2
            p1 /= p1[2, 0]
            p2 /= p2[2, 0]
            cv2.line(frame, (int(p1[0, 0]), int(p1[1, 0])),
                     (int(p2[0, 0]), int(p2[1, 0])), (0, 255, 0), 1)

    # Teken markers
    if ids is not None and len(ids) > 0:
        stag.drawDetectedMarkers(frame, corners, ids)

    # Toon resultaat
    cv2.imshow("Webcam + Grid", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # stop loop
        break
    elif key == ord('c') and ids is not None and len(ids) >= 4 and H is None:
        # Bereken H pas als jij 'c' drukt
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
            print("Homografie berekend!")

cap.release()
cv2.destroyAllWindows()
