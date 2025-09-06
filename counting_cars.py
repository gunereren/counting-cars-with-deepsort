import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


modelName = "yolov8n.pt"
model = YOLO(modelName)
model.to("cuda")
class_names = model.names
min_conf = 0.3

tracker = DeepSort(max_age=1)

cap = cv2.VideoCapture("highway2.mp4")

vehicle_count = 0
counted_ids = set()
fps_list = []
w, h = 0, 0

while True:
    ret,frame = cap.read()
    if not ret:
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            with open("fps_log.txt", "a") as f:
                f.write(f"Avg FPS: {avg_fps:.2f}   Model: {modelName}   Resolution:{w}x{h}   Counted Cars: {vehicle_count}\n")
        break
    #frame = cv2.resize(frame, (640, 360))
    line_y = int(frame.shape[0] * 0.4)
    w = frame.shape[1]
    h = frame.shape[0]
    start = cv2.getTickCount()
    results = model(frame, verbose=False)[0]
    
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls in [2, 3, 5, 7] and conf > min_conf:
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
            
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if cy > line_y - 5 and cy < line_y + 5:
            if track_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_names[track.det_class]}{track_id} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    end = cv2.getTickCount()
    time = (end - start) / cv2.getTickFrequency()
    fps = 1 / time

    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    fps_list.append(fps)
    cv2.imshow("Counting Cars", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()