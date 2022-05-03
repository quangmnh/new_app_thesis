from onnx_helper import ONNXClassifierWrapper
import numpy as np
PRECISION = np.float32
BATCH_SIZE=1
dummy_input_batch = np.zeros((BATCH_SIZE, 48, 48, 1))
N_CLASSES = 5 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("new_model.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
predictions = trt_model.predict(dummy_input_batch)
print(predictions)