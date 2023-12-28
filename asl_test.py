import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model("D:\\Team Project\\ASL\\models\\asl_model.h5")  # Replace with the path to your trained model

# Define the labels corresponding to the classes
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Open the camera (you may need to adjust the camera index)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the frame (resize to the same dimensions as your training data)
    frame = cv2.resize(frame, (100, 100))

    # Expand dimensions to match the model's input shape
    input_data = np.expand_dims(frame, axis=0)

    # Normalize the input data
    input_data = input_data.astype('float32') / 255

    # Make predictions
    predictions = model.predict(input_data)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the predicted class label on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Hand Sign Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
