from model_manager import *
import numpy as np
camera = CameraManagement()
emo_model = ONNXClassifierWrapper("new_model.trt", [1, 5], target_dtype = np.float32)
caffe_model = ONNXClassifierWrapper2("new_caffe.trt", [1, 1, 200, 7], 0.5, target_dtype = np.float32)

print("ready")
while True:
    frame = camera.get_frame()
    box = caffe_model.predict(camera.get_blob(frame))
    if box is None:
        continue
    else:
        (x, y, w, h) = box.astype('int')
        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)
        roi = camera.get_roi(box, frame)
        if roi is None:
            continue
        else:
            label = emo_model.predict(roi)
            print(label)
    cv2.imshow('Emotion :3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break