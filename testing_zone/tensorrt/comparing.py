import cv2
# from model_manager import *

# trt_model = ONNXClassifierWrapper("new_model_fp16.trt", [1, 5], target_dtype = np.float16)
# emo_model = KerasEmotionClassificationModel("./input/facial_emotion_recognition_new_dataset.h5")
# caffe_model = SSDCaffeModel(modelFile="./input/res10_300x300_ssd_iter_140000.caffemodel",configFile="./input/deploy.prototxt.txt")
import os
data_path = ".\input\data"
filepaths = []
images = []
for root, directories, files in os.walk(data_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        image = cv2.imread(filepath)
        label = filepath.split('\\')[4]
        images.append({"label": label, "image": image})
        filepaths.append(filepath)
print(images[0]["label"])
