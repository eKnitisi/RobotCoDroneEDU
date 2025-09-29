import stag
import cv2
import numpy as np

# Marker type instellen
libraryHD = 21

# Definieer de wereldcoördinaten van je grid (meters)
world_points = {
    0: (0, 0),   # Marker ID 00
    1: (2, 0),   # Marker ID 01
    2: (2, 3),   # Marker ID 02
    3: (0, 3)    # Marker ID 03
}

H = None  # Homography matrix

# Webcam openen (0 = standaard camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kan camera niet openen")
    exit()

# ---------- Bereken H één keer ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    (corners, ids, rejected_corners) = stag.detectMarkers(frame, libraryHD)

    if ids is not None and len(ids) >= 4:
        image_pts = []
        world_pts_list = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in world_points:
                c = corners[i][0]
                center = np.mean(c, axis=0)
                image_pts.append(center)
                world_pts_list.append(world_points[marker_id])

        if len(image_pts) == 4:
            H, _ = cv2.findHomography(np.array(image_pts), np.array(world_pts_list))
            print("Homography berekend")
            break

# ---------- Main loop: drone tracking ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    # Hier zou je drone-detectie code komen (bijv. rood object detecteren)
    # Stel drone_pixel = (x, y) de pixelpositie van drone

    # Voorbeeld: dummy drone positie in het midden van beeld
    drone_pixel = (frame.shape[1]//2, frame.shape[0]//2)

    # Zet pixelpositie om naar wereldcoördinaten
    if H is not None:
        pt = np.array([[drone_pixel[0], drone_pixel[1], 1]]).T
        world_pt = H.dot(pt)
        world_pt /= world_pt[2,0]
        drone_world = (world_pt[0,0], world_pt[1,0])
        cv2.circle(frame, drone_pixel, 6, (0,0,255), -1)
        cv2.putText(frame, f"Drone: ({drone_world[0]:.2f}m, {drone_world[1]:.2f}m)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        print(f"Drone wereldpositie: {drone_world}")

    # Teken gridlijnen (optioneel)
    step = 0.5
    for x in np.arange(0, 2.01, step):
        pt1 = np.array([[x,0,1]]).T
        pt2 = np.array([[x,3,1]]).T
        p1 = np.linalg.inv(H).dot(pt1); p2 = np.linalg.inv(H).dot(pt2)
        p1 /= p1[2,0]; p2 /= p2[2,0]
        cv2.line(frame, (int(p1[0,0]), int(p1[1,0])), (int(p2[0,0]), int(p2[1,0])), (0,255,0), 1)

    for y in np.arange(0, 3.01, step):
        pt1 = np.array([[0,y,1]]).T
        pt2 = np.array([[2,y,1]]).T
        p1 = np.linalg.inv(H).dot(pt1); p2 = np.linalg.inv(H).dot(pt2)
        p1 /= p1[2,0]; p2 /= p2[2,0]
        cv2.line(frame, (int(p1[0,0]), int(p1[1,0])), (int(p2[0,0]), int(p2[1,0])), (0,255,0), 1)

    # Teken markers
    if ids is not None and len(ids) > 0:
        stag.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Webcam + Grid + Drone", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
