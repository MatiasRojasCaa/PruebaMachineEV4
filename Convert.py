import tensorflow as tf
import tf2onnx
import onnx

model_path = "mineral_classification_model.keras"
model = tf.keras.models.load_model(model_path)
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]

model.output_names=['output']

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "model" + ".onnx")

import onnxruntime as rt

# Load the ONNX model
sess = rt.InferenceSession("model.onnx")

# Check input and output names
print("Input names:", [input.name for input in sess.get_inputs()])
print("Output names:", [output.name for output in sess.get_outputs()])
