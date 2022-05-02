from onnx_helper import ONNXClassifierWrapper
import numpy as np
PRECISION = np.float32
BATCH_SIZE=32
dummy_input_batch = np.zeros((BATCH_SIZE, 3, 48, 48))
N_CLASSES = 5 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("new_model.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)