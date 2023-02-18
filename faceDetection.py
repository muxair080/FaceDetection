import cv2
import mediapipe as mp 
import time

cap = cv2.VideoCapture('./videos/6.mp4')
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils



faceDetection = mpFaceDetection.FaceDetection()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the size of the display window to match the size of the video frame
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", (width, height))

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read video")
        break
    else:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results =  faceDetection.process(imgRGB)
        # print(results)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width*iw), int(bboxC.height*ih)

                cv2.rectangle(img, bbox, (255,0,255),4)
                cv2.putText(img, f'FPS: {int(detection.score[0]*100)}%',
                (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,255),2)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
    
        cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,0),2)
        cv2.imshow("Image", img)
        cv2.waitKey(10)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()