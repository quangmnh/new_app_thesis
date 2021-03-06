from multiprocessing import dummy
from timeit import timeit
from model_manager import *
import os
from time import time
import psutil
trt_model = ONNXClassifierWrapper("new_model.trt", [1, 5], target_dtype = np.float32)
caffe_model = SSDCaffeModel(modelFile="./input/res10_300x300_ssd_iter_140000.caffemodel",configFile="./input/deploy.prototxt.txt")

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
dummy_input = np.zeros((1, 48, 48, 1))
for _ in range(10):
    _ = trt_model.predict(dummy_input)
for image in images:
    frame = image["image"]
    box = caffe_model.get_boxes(frame=frame, blob=get_blob(frame))
    if box is None:
        continue
    else:
        roi = get_roi(box, frame)
        if roi is None:
            continue
        else:
            start = time()
            label = trt_model.predict(roi)
            time_total+=time()-start
            if label == image["label"]:
                res+=1
            count+=1
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss)
        
    
    # print(emo_model.predict(roi))
print("total time for emotional recognition for tensorrt model: {}".format(time_total*1000))
# print(time_total*1000)    
# print(""res*1.0/count)
print("Total frames processed: ".format(count))
