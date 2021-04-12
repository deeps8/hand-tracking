#importing opencv for image, video etc. 
from cv2 import cv2 as cv
#importing mediapipe for hand tracking
import mediapipe as mp
#time import for fps display
import time

#initializing video capture
cap = cv.VideoCapture(0)

class hand_detector():

    def __init__(self,
                 image_mode=False ,
                 max_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        #initializing the parameters
        self.mode = image_mode
        self.max_hands = max_hands
        self.detection_con = min_detection_confidence
        self.track_con = min_tracking_confidence

        #getting mediapipe hands solution
        self.mp_hands = mp.solutions.hands
        #geting hand module from mediapipe
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_hands,
                                         self.detection_con,
                                         self.track_con)
        #getting drawing utils for drawing lines and dots on hand img
        self.mp_draw = mp.solutions.drawing_utils


    """
        tracking the hands and drawing landmarks on it 
        all 21 points and line joining it
    """
    def find_hands(self,img):
        #converting BGR to RGB img
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        #processing the RGB image and returning processed img with landmarks
        self.results = self.hands.process(imgRGB)

        #check for landmarks
        if(self.results.multi_hand_landmarks):
            #looping thru all landmarks hands (more than 1 hand)
            for handslms in self.results.multi_hand_landmarks:
                #drawing the landarks on image and connecting points too
                self.mp_draw.draw_landmarks(img,handslms,self.mp_hands.HAND_CONNECTIONS)    

        return img

    """
        Function for finding all 21 points on a hand and draw the circles on it
    """
    def find_position(self,img,hand_no=0,draw=True):
        point_list = []
        if(self.results.multi_hand_landmarks):
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for (pid,lm) in enumerate(my_hand.landmark):
                # print(pid,lm)
                (h,w,_) = img.shape
                (cx,cy) = int(lm.x*w),int(lm.y*h)
                # print(pid,cx,cy)
                point_list.append([pid,cx,cy])
                
                if draw:
                    cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)    

        return point_list

def main():

    #for fps display
    ptime = 0
    ctime = 0

    detector = hand_detector(min_detection_confidence=0.6)

    while True:
        #reading image from cam(0)
        (_,img) = cap.read()
    
        img = detector.find_hands(img)
        detector.find_position(img,draw=False)
        #calculating fps
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        #displaying fps on screen/img
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)

        cv.imshow("Image",img)
        cv.waitKey(1)


if(__name__=="__main__"):
    main()