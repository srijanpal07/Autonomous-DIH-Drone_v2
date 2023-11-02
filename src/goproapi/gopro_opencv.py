import cv2
import numpy as np
from goprocam import GoProCamera
from goprocam import constants
import threading,queue

class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # break
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    def release(self):
        self.cap.release()
        return

#cascPath="/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascPath)
gpCam = GoProCamera.GoPro()
#gpCam.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)
#gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.W480)
cap = VideoCapture("udp://127.0.0.1:10000")
# cap = cv2.VideoCapture("udp://127.0.0.1:10000")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    # Capture frame-by-frame
    # ret, frame = cap.read()
    frame = cap.read()

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 #   faces = faceCascade.detectMultiScale(
 #       gray,
 #       scaleFactor=1.1,
 #       minNeighbors=5,
 #       minSize=(30, 30),
 #       flags=cv2.CASCADE_SCALE_IMAGE
 #   )

  #  # Draw a rectangle around the faces
  #  for (x, y, w, h) in faces:
  #      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow("GoPro: %d x %d" % (frame.shape[1], frame.shape[0]),frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
