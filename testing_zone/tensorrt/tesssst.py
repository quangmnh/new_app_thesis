import cv2
# from onnx_helper import ONNXClassifierWrapper
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
print(blob[0].shape)
# trt_model = ONNXClassifierWrapper("new_caffe_fp16.trt",[bruh, bruh2, FACE_NUMBER, BOX] , target_dtype = PRECISION)
# predictions = trt_model.predict(blob)