#Sound Classification on NIGENS Dataset using CNN, KNN, SVM, LSTM and Random Forest

First thing to do:
1. Download the NIGENS dataset: https://zenodo.org/records/2535878
2. In each algorithm's .py file, you can change the dataset path to your dataset's location.
# Path to dataset
data_path = "NigensAudio"

This study examines the performance of several machine learning and deep learning algorithms in classifying sounds from the NIGENS dataset, which consists of 15 unique sound classes. The implementation included five distinct algorithms: Convolutional Neural Networks (CNN), k-Nearest Neighbors (KNN), Support Vector Machines (SVM), Long Short-Term Memory Networks (LSTM), and Random Forest. The models' performances are evaluated using classification measures such as Accuracy, precision, recall, F1-score, and Receiver Operating Characteristic (ROC) curves. The Random Forest method demonstrates the best overall accuracy of 86% when compared to other algorithms, with precision, recall, and F1-score values of 0.88, 0.86, and 0.86. 

Workflow Diagram for this Sound Classification: ![image](https://github.com/user-attachments/assets/a8bb6503-246e-4701-b34c-2e7409772aea)

Tools and Materials Used:
- Librosa: For audio processing and feature extraction.
- TensorFlow/Keras: For implementing the CNN and LSTM models.
- Scikit-learn: Powerful library that can be used to create KNN, SVM, and Random Forest models. It also provides functionality for data preparation and assessment.
- Joblib: Parallel processing tool that speeds up the execution of data processing operations.
- Matplotlib: To view the findings and confusion matrices.

To compare the performance of these five algorithms, we used metrics like F1-score, confusion matrix, recall, accuracy, and precision to evaluate the models.
