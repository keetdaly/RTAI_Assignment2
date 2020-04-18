import cv2
from os import listdir
from os.path import isfile, join

file_path = "results/"
img_array = []

# Creating a list of file names
files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
files = ['frame' + str(i) + ".png" for i in range(1, len(files) + 1)]

# Accessing files in sorted manner to create image array
for i in range(len(files)-1):
    filename = file_path + files[i]
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# writing the video to the file
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 15.0, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()