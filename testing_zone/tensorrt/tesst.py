from onnx_helper import ONNXClassifierWrapper
import numpy as np
PRECISION = np.float32
dummy_input_batch = np.zeros((BATCH_SIZE, 48, 48, 1))
trt_model = ONNXClassifierWrapper("new_model_fp16.trt", [1, 5], target_dtype = np.float16)
predictions = trt_model.predict(dummy_input_batch)
print(predictions)