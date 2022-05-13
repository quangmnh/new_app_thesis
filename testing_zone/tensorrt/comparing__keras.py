from multiprocessing import dummy
from timeit import timeit
from model_manager import *
import os
from time import time
# camera = CameraManagement()
# trt_model = ONNXClassifierWrapper("new_model.trt", [1, 5], target_dtype = np.float32)
emo_model = KerasEmotionClassificationModel("./input/facial_emotion_recognition_new_dataset.h5")
# caffe_model = SSDCaffeModel(modelFile="./input/res10_300x300_ssd_iter_140000.caffemodel",configFile="./input/deploy.prototxt.txt")
trt_model = ONNXClassifierWrapper2("new_caffe.trt",[1, 1, 200, 7] , target_dtype = np.float32)
# print("??????????????")

data_path = "./input/data"
filepaths = []
images = []
res = 0
time_total = 0.0
count =0
count1 = 0
for root, directories, files in os.walk(data_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        # print(filepath)
        image = cv2.imread(filepath)
        label = filepath.split('/')[4]
        # images.append({"label": label, "image": image})
        # count1+=1
        # if count1 == 3:
        #     break
        # frame = image["image"]
        # print("??????????????a")
        blob = get_blob(image)
        box = trt_model.predict(blob)
        if box is None:
            continue
        else:
            # print("??????????????b")
            roi = get_roi(box, image)
            # print("??????????????c")
            # print(roi.shape)
            # time_total += timeit('trt_model.predict(roi)')
            if roi is None:
                continue
            else:
                start = time()
                label_pred = emo_model.predict(roi)
                time_total+=time()-start
                if label == label_pred:
                    res+=1
                count+=1
        
        # filepaths.append(filepath)
# dummy_input = np.zeros((1, 48, 48, 1))
# for _ in range(10):
#     _ = emo_model.predict(dummy_input)

# for image in images:
    
        
    
    # print(emo_model.predict(roi))
print("total time for emotional recognition for tensorrt model: {}".format(time_total*1000))
# print(time_total*1000)    
print("Accuracy: {}".format(res*1.0/count))
print("Total frames processed: {}".format(count))
