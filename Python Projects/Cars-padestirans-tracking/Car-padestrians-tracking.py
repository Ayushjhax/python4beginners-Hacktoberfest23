import cv2
from random import randrange

#our img
# img_file = 'Traffic.jpg'
video = cv2.VideoCapture('carpedestrians.mp4')

#our pretrained car classifier
car_tracker_file = 'cars.xml'

#Create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)

pedestrians_tracker_file = 'haarcascade_fullbody.xml'

pedestrians_tracker = cv2.CascadeClassifier(pedestrians_tracker_file)

#Run forever until car stops or something goes wrong
while True:
    #Read the current frame 
    (read_successful, frame) = video.read() #returns a tuple
#safe config.
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrians_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y), (x+w,y+h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 4)

    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y), (x+w,y+h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 4)

    cv2.imshow('Ayush car detector', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
#Release
video.release()

print("Code Completed")