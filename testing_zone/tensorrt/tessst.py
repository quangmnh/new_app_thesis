import cv2
from onnx_helper import ONNXClassifierWrapper
import numpy as np


frame = cv2.imread("./input/0.png")
PRECISION = np.float16
BATCH_SIZE=1
# dummy_input_batch = np.zeros((BATCH_SIZE, 3, 300, 300))

bruh = 1
bruh2 = 1
FACE_NUMBER = 200
BOX = 7

blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104., 177., 123.))
trt_model = ONNXClassifierWrapper("new_caffe_fp16.trt",[bruh, bruh2, FACE_NUMBER, BOX] , target_dtype = PRECISION)
predictions = trt_model.predict(blob)
# print(predictions)

for i in range(0, predictions.shape[2]):

        confidence = predictions[0, 0, i, 2]

        if confidence > 0.5:
            print(predictions[0, 0, i, :])


modelFile = './input/res10_300x300_ssd_iter_140000.caffemodel'
configFile = './input/deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setInput(blob)
detections = net.forward()
print("bbjdknbjsadnkkkkkkkk")
for i in range(0, detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        print(detections[0, 0, i, :])