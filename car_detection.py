import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
class_names = model.names
cap = cv2.VideoCapture("highway2.mp4")
model.to("cuda")

while True:
    ret,frame = cap.read()

    if not ret:
        break
    start = cv2.getTickCount()
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls in [2] and conf > 0.3:
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_names[cls]} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end = cv2.getTickCount()
    time = (end - start) / cv2.getTickFrequency()
    fps = 1 / time

    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Counting Cars", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()