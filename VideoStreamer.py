import cv2
from ObjectDetection import ObjectDetection

class VideoStreamer(object):
    def read_video(self, file_name):
        """
        The method reads the input video at given rate(fps) and stores the output frames in the given folder.
        :param file_name: The path for the input video
        :return: None
        TODO: Check if we are reading at 30 fps
        """
        # Reading the input video
        video = file_name
        cap = cv2.VideoCapture(video)

        while (cap.isOpened()):
            print("Reading frame")
            # Reading each frame
            ret, frame = cap.read()

            if ret:
                # Resizing the image and providing the same as for input to Yolo model
                height, width, channels = frame.shape
                f = cv2.resize(frame, None, fx=0.4, fy=0.4)
                blob = cv2.dnn.blobFromImage(f, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                # Calling the next module in pipeline to detect frames with cars
                ObjectDetection().detect_car(blob, height, width, frame)
            else:
                break

        # Releasing the video after processing all the frames
        cap.release()
        cv2.destroyAllWindows()
