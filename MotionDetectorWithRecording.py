# Program to implement
# WebCam Motion Detector

import cv2
import pandas 
import time
# importing datetime class from datetime library
from datetime import datetime

from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# assigning our static_back to None (stores the intial data for the screen)
static_back = None

# list when any moving object appear (when any motion is found it will be stored in)
motion_list = [None, None]

# time of movement
time = []

# initializing DataFrame, one column is start  time and other column is end time
df = pandas.DataFrame(columns=["Start", "End"])


video = cv2.VideoCapture(0)


while True:

    check, frame = video.read()

    # initializing motion = 0(no motion)
    motion = 0

    # converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converting gray scale image to GaussianBlur so that change can be detected more efficiently and faster
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # in first iteration we assign the value of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue

    # difference between static background and current frame(which is GaussianBlur)
    # (difference between current frame and background)
    diff_frame = cv2.absdiff(static_back, gray)

    # create a threshold to exclude minute movements
    thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # dialate threshold to further reduce error
    thresh = cv2.dilate(thresh, None, iterations=2)

    # finding contour(variation in the movments ) of moving object
    cnts, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)

    # check each contour
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # we now append status of motion
    motion_list.append(motion)

    # we slice the array from the second last element
    motion_list = motion_list[-2:]

    # appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now().strftime("%H:%M:%S"))

    # appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now().strftime("%H:%M:%S"))

    # displaying image in gray_scale
    cv2.imshow("Gray Frame", gray)

    # displaying the difference in currentframe to the staticframe(very first_frame)
    cv2.imshow("Difference Frame", diff_frame)

    # displaying the black and white image in which if intencity difference greater than 30 it will appear white
    cv2.imshow("Threshold Frame", thresh)

    # displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # if something is movingthen it append the end time of movement
        if motion == 1:
            time.append(datetime.now().strftime("%H:%M:%S"))
        break

# appending time of motion in DataFrame
for i in range(0, len(time), 2):
    df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)

# creating a csv file in which time of movements will be saved


df.to_csv("Time_of_movements.csv")

data=pandas.read_csv('Time_of_movements.csv')

#print(data.Start[0])

#line chart 
plt.plot(data.Start, data.End)

#bar chart
#plt.bar(data.Start, data.End)

plt.xlabel("Motion Start Time")
plt.ylabel("Motion End Time")
plt.title("Motion Graph")
plt.show()




video.release()

# destroying all the windows
cv2.destroyAllWindows()
