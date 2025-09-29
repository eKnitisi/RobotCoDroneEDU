import stag
import cv2
import numpy as np

# Marker type instellen
libraryHD = 21

# Definieer de wereldcoördinaten van je grid (meters)
# Stel dat de markers een rechthoek van 2m x 3m vormen:
world_points = {
    0: (0, 0),   # Marker ID 00
    1: (2, 0),   # Marker ID 01
    2: (2, 3),   # Marker ID 02
    3: (0, 3)    # Marker ID 03
}

H = None  # Homography matrix

# Webcam openen (0 = standaard camera, 1 = externe camera)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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

    if ids is not None and len(ids) >= 4 and H is None:
        image_pts = []
        world_pts = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in world_points:
                c = corners[i][0]  # 4 hoekpunten van marker
                center = np.mean(c, axis=0)  # middelpunt nemen
                image_pts.append(center)
                world_pts.append(world_points[marker_id])

        if len(image_pts) == 4:
            print("4 markers gevonden. Druk op 'c' om homografie te berekenen...")
            key = cv2.waitKey(0) & 0xFF   # wacht tot gebruiker een toets indrukt
            if key == ord('c'):           # alleen bij 'c' berekenen
                H, _ = cv2.findHomography(np.array(image_pts), np.array(world_pts))
                print("Homografie berekend!")

    # Als homography bekend is → grid tekenen
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
            cv2.line(frame, (int(p1[0, 0]), int(p1[1, 0])), (int(p2[0, 0]), int(p2[1, 0])), (0, 255, 0), 1)

        # horizontale lijnen
        for y in np.arange(0, 3.01, step):
            pt1 = np.array([[0, y, 1]]).T
            pt2 = np.array([[2, y, 1]]).T
            p1 = H_inv @ pt1
            p2 = H_inv @ pt2
            p1 /= p1[2, 0]
            p2 /= p2[2, 0]
            cv2.line(frame, (int(p1[0, 0]), int(p1[1, 0])), (int(p2[0, 0]), int(p2[1, 0])), (0, 255, 0), 1)

    # Teken markers
    if ids is not None and len(ids) > 0:
        stag.drawDetectedMarkers(frame, corners, ids)

    # Toon resultaat
    cv2.imshow("Webcam + Grid", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
