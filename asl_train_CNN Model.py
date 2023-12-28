import os
import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_root = "D:\\Team Project\\ASL\\dataset\\Source"

asl_images = []
asl_labels = []

# Iterate through the subdirectories, where each subdirectory corresponds to a letter (e.g., "A", "B", etc.)
for letter_dir in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root, letter_dir)):
        letter_label = letter_dir  # Use the directory name as the label

        # Iterate through the images in the letter directory
        for filename in os.listdir(os.path.join(dataset_root, letter_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(dataset_root, letter_dir, filename)

                # Read the image
                image = cv2.imread(image_path)

                # Resize the image to a smaller resolution, e.g., (100, 100)
                image = cv2.resize(image, (100, 100))

                # Append the image to the list
                asl_images.append(image)

                # Append the label (letter) to the labels list
                asl_labels.append(letter_label)

# Convert the lists to numpy arrays
asl_images = np.array(asl_images)

# Define a mapping from string labels to numerical labels
label_mapping = {letter: idx for idx, letter in enumerate(np.unique(asl_labels))}

# Convert the string labels to numerical labels
numerical_labels = np.array([label_mapping[label] for label in asl_labels])

# Convert numerical labels to one-hot encoding
asl_labels = to_categorical(numerical_labels)

# Preprocess the images
asl_images = asl_images.astype('float32') / 255

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(asl_images, asl_labels, test_size=0.2, random_state=42)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))  # 26 units for 26 classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# Save the trained model to an HDF5 file
model.save("D:\\Team Project\\ASL\\models\\asl_model.h5")


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")