Lung Image Classification with Convolutional Neural Networks
This Jupyter Notebook demonstrates the process of building a Convolutional Neural Network (CNN) to classify lung images into three categories: lung_aca, lung_n, and lung_scc. The notebook covers data preparation, model building, training, and evaluation.
Prerequisites
Before running the notebook, ensure you have the necessary dependencies installed. You can install them using the following command:
txt
Notebook Overview
1. Install Dependencies
The notebook begins by installing the required Python packages using pip. This includes libraries such as TensorFlow, scikit-learn, OpenCV, and others necessary for image processing and model training.
2. Import Dependencies
The necessary libraries are imported, including NumPy, Pandas, Matplotlib, TensorFlow, Keras, and others. These libraries are essential for data manipulation, visualization, and building the CNN model.
3. Define Path to Images
The path to the dataset containing lung images is defined. The dataset is organized into subdirectories, each representing a class.
4. Display Sample Images
The notebook displays three random sample images from each class to provide a visual understanding of the dataset.
5. Define Model Hyperparameters
Key hyperparameters for the model are defined, including image size, train-validation split ratio, number of epochs, and batch size.
6. Load and Preprocess Images
Images are loaded and preprocessed. They are resized to a uniform size, and labels are one-hot encoded. The dataset is then split into training and validation sets.
7. Build the CNN Model
A sequential CNN model is constructed using Keras. The model consists of convolutional layers, max-pooling layers, dense layers, and dropout layers to prevent overfitting.
8. Compile the Model
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. Accuracy is used as the evaluation metric.
9. Define Callbacks
Callbacks for early stopping and learning rate reduction are defined to enhance the training process. A custom callback is also implemented to stop training once a validation accuracy of 90% is achieved.
10. Train the Model
The model is trained on the training dataset, with validation on the validation dataset. The training process is monitored using the defined callbacks.
11. Evaluate the Model
The training history is visualized, showing the loss and accuracy over epochs. The model's performance is evaluated using a confusion matrix and classification report.
12. Results
The notebook concludes with the display of a confusion matrix and a classification report, providing insights into the model's performance on the validation dataset.
Conclusion
This notebook provides a comprehensive guide to building a CNN for image classification tasks. By following the steps outlined, you can adapt the process to other image classification problems with similar datasets.