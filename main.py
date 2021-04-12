import cv2
from tracker import *
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("Vid_obj_detect.mp4")

#Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)
    #Extract region on interest
    #roi = frame[340:600,200: 1200] # Use for perticular frame
    #1.Object detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 100,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detection = []

    for cnt in contours:
        #Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(frame, [cnt], -1, (0,255, 0), 2)
            x, y ,w ,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            detection.append([x,y,w,h])

    #2. Object tracking
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        x,y,w,h, id = box_id
        cv2.putText(frame, str(id),  (x,y-15), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    #cv2.imshow('ROI', roi)
    cv2.imshow('Frame',frame)
    cv2.imshow('Mask', mask)
    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()