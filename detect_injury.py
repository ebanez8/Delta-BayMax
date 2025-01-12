import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("final_injury_illness_model.keras")

# Define the class labels (this must match your dataset classes)
class_labels = ['Abrasions', 'Bruises',"Burns","Cuts"]  # Replace with your actual class names

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the image to the same size as the model input (224x224 for MobileNetV2)
    img = cv2.resize(frame, (224, 224))

    # Convert image to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_name = class_labels[predicted_class[0]]

    # Display the result
    cv2.putText(frame, f"Prediction: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()