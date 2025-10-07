import cv2
import time

counter = 1
last_update = time.time()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kan de webcam niet openen")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen")
        break

    # Bereken of er een seconde is verstreken om de teller te verhogen
    if time.time() - last_update >= 1:
        print("Teller:", counter)
        counter += 1
        last_update = time.time()

    cv2.imshow("Webcam", frame)

    # Stoppen bij 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programma gestopt.")
