import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time

# model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite')
# model = YOLO('yolov8n_int8_edgetpu.tflite')
# model = YOLO('yolov8n_custom_200_epoches_CPU_510_images_int8_edgetpu.tflite')
# model = YOLO('yolov8n_custom_200_epoches_CPU_510_images_int8_default_edgetpu.tflite')
# model = YOLO('yolo11n_full_integer_quant_edgetpu.tflite')
# model = YOLO('yolo11n_custom_200_epoches_CPU_510_images_imgsz_320_int8_edgetpu.tflite') # works!
# model = YOLO('yolo11n_custom_200_epoches_CPU_510_images_imgsz_320_int8.tflite') # img size mismatch
# model = YOLO('yolo11n_custom_200_epoches_CPU_510_images_imgsz_320_full_integer_quant_edgetpu.tflite') # works
model = YOLO('yolov8n_custom_200_epoches_CPU_510_images_imgsz_256_full_integer_quant_edgetpu.tflite')

cap = cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    results = model.predict(frame, iou=0.2, conf=0.4, imgsz=256) #  iou=0.01, conf=0.8, 256)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        # print(f"d: {d}, class_list length: {len(class_list)}")
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    # Display FPS on frame
    cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1,1)

    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
