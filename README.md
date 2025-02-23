# Lung Image Classification with Convolutional Neural Networks (CNN)

## Overview
This Jupyter Notebook demonstrates the process of building a **Convolutional Neural Network (CNN)** to classify lung images into three categories:
- **lung_aca** (lung adenocarcinoma)
- **lung_n** (normal lung tissue)
- **lung_scc** (lung squamous cell carcinoma)

The notebook covers **data preparation, model building, training, and evaluation** to develop an effective classification model.

---
## Prerequisites
Before running the notebook, ensure you have the necessary dependencies installed. Install them using the following command:
```bash
pip install -r requirements.txt
```

---
## Notebook Workflow

### 1. Install Dependencies
The required Python libraries are installed using `pip`. These include:
- **TensorFlow** and **Keras** for building the CNN model.
- **scikit-learn** for preprocessing and evaluation.
- **OpenCV** for image processing.
- **NumPy, Pandas, and Matplotlib** for data handling and visualization.

### 2. Import Dependencies
Necessary libraries are imported for data manipulation, visualization, and CNN model construction.

### 3. Define Path to Images
The dataset containing lung images is organized into subdirectories, each corresponding to a class (**lung_aca, lung_n, lung_scc**).

### 4. Display Sample Images
Three random images from each class are displayed to provide a visual overview of the dataset.

### 5. Define Model Hyperparameters
Key hyperparameters for training the CNN model are defined, including:
- **Image size** for input consistency.
- **Train-validation split ratio**.
- **Number of epochs** and **batch size**.

### 6. Load and Preprocess Images
- Images are loaded and resized to a uniform size.
- Labels are one-hot encoded for classification.
- The dataset is split into **training** and **validation** sets.

### 7. Build the CNN Model
A **Sequential CNN model** is built using Keras. The architecture includes:
- **Convolutional layers** for feature extraction.
- **Max-pooling layers** for downsampling.
- **Dense layers** for classification.
- **Dropout layers** to prevent overfitting.

### 8. Compile the Model
- The model is compiled using the **Adam optimizer**.
- **Categorical cross-entropy** is used as the loss function.
- **Accuracy** is selected as the evaluation metric.

### 9. Define Callbacks
To optimize training, the following callbacks are implemented:
- **Early Stopping**: Stops training if validation loss stops improving.
- **Learning Rate Reduction**: Adjusts learning rate dynamically.
- **Custom Callback**: Stops training once validation accuracy reaches **90%**.

### 10. Train the Model
- The model is trained on the **training dataset** with validation on the **validation dataset**.
- Training progress is monitored using defined callbacks.

### 11. Evaluate the Model
- The training history is visualized, showing **loss** and **accuracy** over epochs.
- The model's performance is analyzed using a **confusion matrix** and **classification report**.

---
## Results
- A **confusion matrix** and **classification report** summarize model performance.
- The results provide insights into accuracy, precision, recall, and F1-score across all three classes.

---
## Conclusion
This notebook serves as a comprehensive guide for building a CNN-based image classification model. By following the outlined steps, the process can be adapted for **other image classification tasks** with similar datasets.

For improvements, consider:
- **Data augmentation** to enhance generalization.
- **Hyperparameter tuning** for better optimization.
- **Transfer learning** with pre-trained CNN architectures for improved accuracy.

---
## License
This project is open-source and available for use and modification under the **MIT License**.
