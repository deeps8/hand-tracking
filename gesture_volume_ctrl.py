from cv2 import cv2 as cv
import mediapipe as mp
import numpy as nm
import hand_tracking_module as htm
import time
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


volume_range = volume.GetVolumeRange()

min_volume = volume_range[0]
max_volume = volume_range[1]

ptime = 0
vol = 0
vol_bar = 400
per = 0

cap = cv.VideoCapture(0)
cap.set(3,620)
cap.set(4,480)

detector = htm.hand_detector(min_detection_confidence=0.7)

while True:
    (_,img) = cap.read()

    img = detector.find_hands(img)

    lmlist = detector.find_position(img,draw=False)
    # print(lmlist)
    if(len(lmlist)!=0):
        # for hand gesture for volume control we need only 2 points [4,8]
        (x1,y1) = (lmlist[4][1],lmlist[4][2])
        (x2,y2) = (lmlist[8][1],lmlist[8][2])
        (cx,cy) = int((x1+x2)/2),int((y1+y2)/2)

        cv.circle(img,(x1,y1),10,(255,0,255),cv.FILLED)
        cv.circle(img,(x2,y2),10,(255,0,255),cv.FILLED)
        cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255,0,255),3)

        distance = math.hypot(x1-x2,y1-y2)
        # print(distance)
        if(distance<=40):
            cv.circle(img,(cx,cy),10,(0,255,0),cv.FILLED)
        
        vol = nm.interp(distance,[40,250],[min_volume,max_volume])
        vol_bar = nm.interp(distance,[40,250],[400,150])
        per = nm.interp(distance,[40,250],[0,100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
    
    cv.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(img,(50,int(vol_bar)),(85,400),(0,255,0),cv.FILLED)
    cv.putText(img,str(f"{int(per)} %"),(50,420),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)

    #fps
    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    cv.putText(img,str(f"FPS : {fps}"),(10,30),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)

    cv.imshow("Image",img)
    cv.waitKey(1)
    