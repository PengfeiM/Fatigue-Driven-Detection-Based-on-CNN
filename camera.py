import cv2
import time

cap=cv2.VideoCapture('G:\\ustc\\bishe\\Captures\\002.WMV')
while cap.isOpened():
	ret,frame=cap.read()
	cv2.imshow('capture', frame)
	time.sleep(0.050)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		 break
cap.release()
cv2.destroyAllWindows()