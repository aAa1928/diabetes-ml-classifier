# diabetes-ml-classifier

PyTorch deep learning model to diagnostically predict whether a patient has diabetes. Data from National Institute of Diabetes and Digestive and Kidney Diseases.

[Dataset link](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/)

## Overview

This project uses PyTorch to build a neural network classifier for diabetes prediction. The model analyzes several medical predictor variables to determine if a patient has diabetes.

## Dataset

The dataset includes diagnostic measurements such as:

- Pregnancies
- Glucose levels
- Blood pressure
- Skin thickness
- Insulin levels
- BMI
- Diabetes pedigree function
- Age

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- numpy

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script: `python main.py`

## Model Architecture

- Input layer: 8 features
- Hidden layers: 2 fully connected layers
- Output layer: Binary classification (0: No diabetes, 1: Diabetes)

## Results

The model achieves approximately 78% accuracy on the test set.

## License

MIT License
