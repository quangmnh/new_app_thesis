import onnx

# Preprocessing: load the ONNX model
model_path = './new_model.onnx'
onnx_model = onnx.load(model_path)

# print('The model is:\n{}'.format(onnx_model))

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')