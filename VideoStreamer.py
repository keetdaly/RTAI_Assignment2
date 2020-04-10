import cv2
from object_detection import detect_car

def read_video(file_name):
    """
    The method reads the input video at given rate(fps) and stores the output frames in the given folder.
    :param file_name: The path for the input video
    :return: None
    """
    # Start video
    video = file_name
    cap = cv2.VideoCapture(video)
    frame_predictions = []

    while (cap.isOpened()):
        print("Reading frame")
        # Read frame
        ret, frame = cap.read()

        if ret:
            height, width, channels = frame.shape

            # Resize image, process for input to Yolo
            f = cv2.resize(frame, None, fx=0.4, fy=0.4)
            blob = cv2.dnn.blobFromImage(f, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            detect_car(blob, height, width)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    with open("frame_predictions.txt", "w") as f:
        for l in frame_predictions:
            f.write(str(l))
            f.write('\n')