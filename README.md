# ASL Hand Sign Recognition - Model Training and Testing

This repository contains the code for training a Convolutional Neural Network (CNN) model for American Sign Language (ASL) hand sign recognition. The trained model can be used to make real-time predictions using a webcam.

## Prerequisites

- Python 3.x
- Keras
- OpenCV
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
   git clone https://github.com/blessingstonsk/ASL-Recognition-with-Deep-Learning.git
   cd your-repository

2. Install the required dependencies:

   pip install -r requirements.txt

## Dataset

Download the ASL Alphabet dataset from Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Extract the dataset and organize it in the following structure:

```
dataset/
|-- A/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- B/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- ...
|-- Z/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
```

## Model Training

1. Modify the `dataset_root` variable in `ASL_Train_CNN_Model.py` to the path where your ASL dataset is stored.

2. Run the training script:

   python ASL_Train_CNN_Model.py

3. The trained model will be saved as `asl_model.h5` in the specified directory.

## Model Testing

1. Modify the path to the trained model in `ASL_Test.py`:

   ```python
   model = load_model("path/to/asl_model.h5")
   ```

2. Run the testing script:

   python ASL_Test.py

3. The webcam will open, and the real-time predictions for ASL hand signs will be displayed.

## Important Notes

- Ensure that you have the required Python libraries installed by running the `pip install -r requirements.txt` command.
- Adjust the paths in the scripts according to your dataset and file locations.
- The model is trained for a specific class ordering (A to Z). Make sure your dataset follows the same ordering.

Feel free to reach out for any questions or improvements!
