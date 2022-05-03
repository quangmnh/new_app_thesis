from model_manager import *
import os
# camera = CameraManagement()
trt_model = ONNXClassifierWrapper("new_model_fp16.trt", [1, 5], target_dtype = np.float16)
emo_model = KerasEmotionClassificationModel("./input/facial_emotion_recognition_new_dataset.h5")
caffe_model = SSDCaffeModel(modelFile="./input/res10_300x300_ssd_iter_140000.caffemodel",configFile="./input/deploy.prototxt.txt")
# print("??????????????")

data_path = "./input/data1"
filepaths = []
images = []
for root, directories, files in os.walk(data_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        print(filepath)
        image = cv2.imread(filepath)
        label = filepath.split('/')[4]
        images.append({"label": label, "image": image})
        filepaths.append(filepath)

for image in images:
    frame = image["image"]
    # print("??????????????a")
    box = caffe_model.get_boxes(frame=frame, blob=get_blob(frame))
    # print("??????????????b")
    roi = get_roi(box, frame)
    # print("??????????????c")
    print(box)
    print(emo_model.predict(roi))
    print(trt_model.predict(roi)[0])
    

