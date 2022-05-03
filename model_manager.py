from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import *
import numpy as np
import cv2

class KerasEmotionClassificationModel():
    __class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    def __init__(self, model_path, display = False, GPU = True):
        self.model = load_model(model_path)
        self.display = display
    def predict(self, frame, box, gray_frame=None):
        if gray_frame is None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = box.astype('int')

        gray_roi = gray_frame[y:h, x:w]

        if gray_roi.shape[0] == 0 or gray_roi.shape[1] == 0:
            return None
        else:
            gray_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([gray_roi]) != 0:
            roi = gray_roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            predict = self.model.predict(roi)[0]
            if self.display:
                print(self.model.predict(roi))
            label = self.__class_labels[predict.argmax()]
            if self.display:
                print(label)
            return label

        if self.display:
            label_position = (x, y - 20)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        if self.display:
            cv2.imshow('Emotion :3', frame)
        return None
class SSDCaffeModel():
    def __init__(self, confidence_threshold = 0.5, modelFile = 'res10_300x300_ssd_iter_140000.caffemodel', configFile = 'deploy.prototxt.txt', display = False):
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.conf_threshold = confidence_threshold
    def get_boxes(self, frame, gray_frame = None):
        res = []
        if gray_frame is None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104., 177., 123.))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:

                # Face bounding box
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                res.append(box)
                # (x, y, w, h) = box.astype('int')
                # cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)
        return res

class 
