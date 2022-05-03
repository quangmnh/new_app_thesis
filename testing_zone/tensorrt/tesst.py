from onnx_helper import ONNXClassifierWrapper
import numpy as np
PRECISION = np.float32
trt_model = ONNXClassifierWrapper("new_model_fp16.trt", [1, 5], target_dtype = np.float16)
predictions = trt_model.predict(dummy_input_batch)
print(predictions)