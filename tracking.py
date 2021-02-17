import cv2

# our image
img_file = 'C:/Users/DELL/Desktop/car ped detection/road.jpeg'
video = cv2.VideoCapture(
    'C:/Users/DELL/Desktop/car ped detection/carvideo.mp4')


# pretrained car classifier and pedestrian classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


# create car classifier

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# run forever untill car stops

while True:
    read_successful, frame = video.read()

    # safe coding
    if read_successful:
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars(coordinates) and pedestrians
    cars = car_tracker.detectMultiScale(greyscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)
    # print(cars)

    # draw rectangles around cars
    for x, y, w, h in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
# draw rectangles around pedestrians
    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the videcapture object
video.release()


"""


# create opencv image
img = cv2.imread(img_file)


# create classifier file
car_tracker = cv2.CascadeClassifier(classifier_file)


# convert image into BLACK AND WHITE

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# DETECT CARS (coordinates)
cars = car_tracker.detectMultiScale(black_n_white)
print(cars)


# draw rectangles around cars

for x, y, w, h in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# display image with faces spotted
cv2.imshow('photos-road', img)
cv2.waitKey()

"""

print("code completed")
