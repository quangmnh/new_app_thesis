from multiprocessing import dummy
from timeit import timeit
from model_manager import *
import os
from time import time
import psutil
# camera = CameraManagement()
# trt_model = ONNXClassifierWrapper("new_model.trt", [1, 5], target_dtype = np.float32)
# emo_model = KerasEmotionClassificationModel("./input/facial_emotion_recognition_new_dataset.h5")
# caffe_model = SSDCaffeModel(modelFile="./input/res10_300x300_ssd_iter_140000.caffemodel",configFile="./input/deploy.prototxt.txt")
# print("??????????????")
trt_model = ONNXClassifierWrapper2("new_caffe.trt",[1, 1, 200, 7] , target_dtype = np.float32)

data_path = "./input/data"
filepaths = []
images = []
res = 0
time_total = 0.0
count =0
for root, directories, files in os.walk(data_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        # print(filepath)
        image = cv2.imread(filepath)
        label = filepath.split('/')[4]
        images.append({"label": label, "image": image})
        filepaths.append(filepath)
dummy_input = np.zeros((1, 3, 300, 300))
for _ in range(10):
    _ = trt_model.predict(dummy_input)
for image in images:
    frame = image["image"]
    # print("??????????????a")
    blob = get_blob(frame)
    start = time()
    box = trt_model.predict(blob)
    # box = caffe_model.get_boxes(frame=frame, blob=get_blob(frame))
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
    time_total+=time()-start
    count+=1
    # print(emo_model.predict(roi))
print("Total time for caffe tensorrt model: {}".format(time_total*1000))   
# print(res*1.0/count)
print("Framws processed: {}".format(count))
exit()