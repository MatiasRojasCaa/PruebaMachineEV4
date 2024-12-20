import onnxruntime as rt
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the ONNX model
sess = rt.InferenceSession("model.onnx")

# Prepare the input image (e.g., resizing it to 224x224 and making sure it has 3 channels)
img_path = "Cobre (1).jpg"  # Provide the correct path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
img_array = image.img_to_array(img)  # Convert image to array

# Add a batch dimension (for example, shape: (1, 224, 224, 3))
img_array = np.expand_dims(img_array, axis=0)

# If needed, normalize or preprocess the image (e.g., by scaling it)
img_array = img_array.astype(np.float32)
img_array /= 255.0  # Assuming your model was trained with scaled images

# Run the model on the input image
input_name = 'digit'  # Input name from the model
output_name = 'output'  # Output name from the model
predictions = sess.run([output_name], {input_name: img_array})

# Debugging output to check predictions shape and values
print(f"Predictions shape: {predictions[0].shape}")
print(f"Predictions: {predictions[0]}")

# Define class labels (replace with actual class names used for training)
class_labels = ["Class1", "Class2", "Class3", "Class4", "Class5"]

# Get the predicted class index
predicted_class_idx = np.argmax(predictions[0])  # Index of the maximum probability

# Output the predicted class and its probability
print(f"Predicted Class: {class_labels[predicted_class_idx]}")
print(f"Probability: {predictions[0][0][predicted_class_idx]}")

