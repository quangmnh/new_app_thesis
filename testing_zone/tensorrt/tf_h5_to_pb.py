import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('input/facial_emotion_recognition_new_dataset.h5')
#tf.saved_model.save(model, "model/facial_emotion_recognition_new_dataset")
#onnx_model = keras2onnx.convert_keras(model, model.name)
#onnx.save_model(onnx_model, "facial_emotion_recognition_new_dataset.onnx")

print(model.get_weights())