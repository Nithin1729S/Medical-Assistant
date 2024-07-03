# Medical Assistant

This Streamlit application provides a user interface for various medical diagnostics and classifications.


## Demo


https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/f7660a23-4d62-4b77-904d-8af3eb2b9b32





[Watch the Demo Video on YouTube](https://youtu.be/b8vmy75NC7w)
## Screenshots

![Screenshot from 2024-07-04 03-59-44](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/84a2044d-0842-4510-9584-36c9fe046753)

![Screenshot from 2024-07-04 03-59-50](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/7a4d0c15-68d7-4454-8db0-eb6e0cfd7d4c)
![Screenshot from 2024-07-04 03-59-53](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/025aa5d0-5cc9-40d2-998f-5f2c521eba7b)


![Screenshot from 2024-07-04 03-59-55](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/1b8bd03d-8993-4bab-b710-1d5db671e3f0)
![Screenshot from 2024-07-04 04-00-06](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/f4aaec4e-582f-4c86-be87-2d61fd2545b1)
![Screenshot from 2024-07-04 04-15-43](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/327d579f-ab2c-4add-b1e9-d4769bc630af)


![Screenshot from 2024-07-04 04-02-18](https://github.com/Nithin1729S/Medical-Assistant/assets/78496667/7d124617-e1c9-4ecf-8691-89be7e9d51fa)



## Live 

[Live](https://nithin1729s-medical-assistant-main-7hyyuw.streamlit.app/)


## Overview

This application uses Streamlit for creating a web-based interface. It includes multiple pages for different medical purposes:
- Home
- Heart Disease Prediction
- Tuberculosis Detection
- Skin Cancer Classification

Each page corresponds to a specific diagnostic or classification task related to medical conditions.

## Heart Disease Prediction

#### Data Collection and Preprocessing

The heart disease prediction page utilizes a dataset from Cleveland Clinic Foundation for heart disease. Data exploration, cleaning, and preprocessing involve:
- Loading and inspecting the dataset to understand features like age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol levels (`chol`), etc.
- Exploratory data analysis (EDA) to visualize distributions, correlations, and class imbalances within the dataset.

#### Model Training

Models such as Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest are trained on the preprocessed data:
- Implementation of various classifiers using scikit-learn.
- Model evaluation using metrics like accuracy, precision, recall, and F1-score.
- Hyperparameter tuning using techniques like RandomizedSearchCV and GridSearchCV to optimize model performance.

#### Model Deployment and Visualization

- The selected model (e.g., Logistic Regression) is deployed using Streamlit for real-time predictions.
- Visualizations include ROC curves, confusion matrices, and feature importance plots to interpret model decisions.


## Skin Cancer Prediction using Convolutional Neural Networks


1. **Dataset Preparation:**
   - The HAM10000 dataset consists of skin lesion images categorized into seven types of skin diseases.
   - Each image is labeled with a diagnosis code (`dx`) representing the type of skin disease.

2. **Image Loading and Preprocessing:**
   - Images are loaded from the dataset directory using Python's `glob` module.
   - Metadata from the CSV file (`HAM10000_metadata.csv`) is used to map image paths and disease labels.
   - Images are resized to a standard size and normalized for model training.

3. **Model Building:**
   - A Convolutional Neural Network (CNN) architecture is constructed using Keras.
   - The model consists of multiple convolutional layers followed by max-pooling layers for feature extraction.
   - Batch normalization and dropout layers are used to improve generalization and prevent overfitting.
   - The final layer uses softmax activation to output probabilities for each disease class.

4. **Training:**
   - The dataset is split into training and validation sets for model training and evaluation.
   - Data augmentation techniques (e.g., rotation, zoom, flip) are applied to increase dataset variability.
   - The model is trained using the Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`).
   - Training progress and metrics (e.g., loss, accuracy) are monitored and visualized using callbacks.

5. **Evaluation and Testing:**
   - Model performance is evaluated on the validation set using metrics like accuracy and confusion matrix.
   - Test-time augmentation is used to improve predictions by averaging predictions of multiple augmentations of each test image.

6. **Deployment:**
   - Once trained and evaluated, the model can be deployed in various environments (e.g., local deployment, cloud services) for inference.


## Tuberculosis Detection using Convolutional Neural Networks

This repository contains code for training a Convolutional Neural Network (CNN) to detect tuberculosis from chest radiography images using the TB Chest Radiography Database.

## Dataset and Splitting

- The TB Chest Radiography Database is used for training, testing, and validation.
- Data is split into train (70%), test (15%), and validation (15%) sets using `splitfolders`.

## Data Preprocessing and Augmentation

Images are preprocessed and augmented using `ImageDataGenerator` from Keras:
- Rescaled to [0, 1]
- Applied shear, zoom, and horizontal flip for augmentation.

## Model Architecture

A pre-trained DenseNet169 model from ImageNet is used as a feature extractor:
- Freezing all layers of DenseNet to retain pre-trained weights.
- Added additional layers for classification:
  - Flatten layer
  - Dense layer with 256 units and ReLU activation
  - Dropout layer with 50% dropout rate
  - Output layer with 2 units and softmax activation for binary classification (TB vs Non-TB).

## Training and Evaluation

- Model is compiled with categorical cross-entropy loss and Adam optimizer.
- Early stopping and model checkpointing are used for training monitoring and saving the best model.
- Trained for 10 epochs with batch size of 16 on the training set.

## Evaluation on Test Set

- Achieved validation accuracy of 99.06% after early stopping.
- Evaluated on the test set, achieving an accuracy of 98.96%.

## Convert to TFLite Format

- Model is converted to TensorFlow Lite (TFLite) format for deployment on mobile and edge devices:
- Optimized with default optimizations for size and speed.



## Installation and Usage

1. Clone the repository:
```shell
git clone https://github.com/Nithin1729S/Medical-Assistant.git
cd Medical-Assistant
```

2. Install the required dependencies: 
```shell
pip install -r requirements.txt
```

3. Run the command: 
```shell
streamlit run main.py
```



