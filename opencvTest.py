import stag
import cv2

# Marker type instellen
libraryHD = 21

# Webcam openen (0 = standaard camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kan camera niet openen")
    exit()

while True:
    # Lees een frame van de webcam
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    # Detecteer markers in het frame
    (corners, ids, rejected_corners) = stag.detectMarkers(frame, libraryHD)

    # Als er markers gevonden zijn, teken ze in beeld
    if ids is not None and len(ids) > 0:
        stag.drawDetectedMarkers(frame, corners, ids)

    # Toon het frame in een venster
    cv2.imshow("Webcam + Stag Markers", frame)

    # Stop met 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Netjes afsluiten
cap.release()
cv2.destroyAllWindows()
