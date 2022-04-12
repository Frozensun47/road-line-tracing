import cv2
import numpy as np

my_video =cv2.VideoCapture('road_car_view.mp4')

while True:
    ret,original_frame=my_video.read()
    if not ret :
        my_video =cv2.VideoCapture('raod_car_view.mp4')
        continue
    frame=cv2.GaussianBlur(original_frame,(5,5),0)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_value=np.array([18,85,140])
    upper_value=np.array([55,255,255])
    
    mask=cv2.inRange(hsv,lower_value,upper_value)
    edges=cv2.Canny(mask,74,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),2)
    
    cv2.imshow("frame",frame)
    cv2.imshow("edges",edges)
    key=cv2.waitKey(25)
    if(key==27):
        break
    


my_video.release()
cv2.destroyAllWindows()
