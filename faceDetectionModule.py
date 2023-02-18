import cv2
import mediapipe as mp 
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()
    
    def findFaces(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =  self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
              
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width*iw), int(bboxC.height*ih)

                bboxs.append([id,bbox, detection.score])
                if draw:
                     img = self.fancyDraw(img, bbox)
                     cv2.putText(img, f'FPS: {int(detection.score[0]*100)}%',
                     (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                    3,(255,0,255),2)

        return img,bboxs


    def fancyDraw(self, img, bbox, l=30, t=10, rt=3):
        x,y,w,h = bbox
        x1,y1, = x+w, y+h 
        cv2.rectangle(img, bbox, (255,0,255),rt)
        # Top Left Cornor
        cv2.line(img, (x,y), (x+l,y),(255,0,255),t)
        cv2.line(img, (x,y), (x,y+l),(255,0,255),t)

           # Top Right Cornor
        cv2.line(img, (x1,y), (x1-l,y),(255,0,255),t)
        cv2.line(img, (x1,y), (x1,y+l),(255,0,255),t)
        
           # Bottom Left Cornor
        cv2.line(img, (x,y1), (x+l,y1),(255,0,255),t)
        cv2.line(img, (x,y1), (x,y1-l),(255,0,255),t)

           # Bottom Right Cornor
        cv2.line(img, (x1,y1), (x1-l,y1),(255,0,255),t)
        cv2.line(img, (x1,y1), (x1,y1-l),(255,0,255),t)
        return img

    
        

 

def main():
    cap = cv2.VideoCapture('./videos/10.mp4')
    pTime = 0

    detector = FaceDetector()

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
            img, bboxs = detector.findFaces(img)
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
            
            cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,
                        3,(255,0,0),2)
            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main();