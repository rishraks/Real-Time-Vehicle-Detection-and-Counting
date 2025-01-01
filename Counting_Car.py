import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np

# Video
cap = cv2.VideoCapture("./Videos/cars.mp4")

# Model
model = YOLO("./YOLO_Weights/yolov8l.pt")

coco_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Mask
mask = cv2.imread("./Images/mask.png")

# Tracker
tracker = Sort(max_age = 20, min_hits = 1, iou_threshold = 0.3)

tracking_line = [250, 360, 673, 360]
count = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        mask_region = cv2.bitwise_and(frame, mask)
        results = model(mask_region, stream= True)
        detections = np.empty((0,5))
        
        for r in results:
            boxes =r.boxes
            for box in boxes:
                
                # Bounding Boxes
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1, y2-y1
                bbox = int(x1), int(y1), int(w), int(h)
                
                # Confidence Score
                confidence = math.ceil((box.conf[0]*100))/100

                # Class Name
                cls = int(box.cls[0])
                current_class = coco_classes[cls]
                
                if current_class == "car" or current_class == "truck" or current_class == "bus" or current_class == "motorbike" and confidence >0.1:
                    current_array = np.array([x1,y1,x2,y2,confidence])
                    detections = np.vstack((detections,current_array))
        
        results_tracker = tracker.update(detections)
        cv2.line(frame, (tracking_line[0],tracking_line[1]),(tracking_line[2],tracking_line[3]), (255,0,255),4)
        
        for result in results_tracker:
            x1,y1,x2,y2,Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(frame, bbox, l=4,t=2)
            cvzone.putTextRect(frame, f"{Id}", (max(0,x1),max(40,y1)), scale=0.7, thickness=1,offset=2)
            
            cx,cy = x1+w//2, y1+h//2
            cv2.circle(frame, (cx,cy), 4, (0,0,255),-1)

            if tracking_line[0] <cx< tracking_line[2] and tracking_line[1] -15 < cy < tracking_line[3]+15:
                if count.count(Id) == 0:
                    count.append(Id)
                    cv2.line(frame, (tracking_line[0],tracking_line[1]),(tracking_line[2],tracking_line[3]), (0,255,0),4)
                    
        cvzone.putTextRect(frame, f'Count:{len(count)}',(50,50),scale =2, thickness=2, offset=5)    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()