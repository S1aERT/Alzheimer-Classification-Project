**Early Detection of Alzheimer’s Disease Using Deep Learning**

This project focuses on the early detection and classification of Alzheimer’s disease using deep learning techniques applied to brain MRI images. It was developed as part of the Project Based Learning – II (PBL-II) course in the Bachelor of Technology (B.Tech) Computer Science & Engineering program at Symbiosis Institute of Technology, Pune.

The primary goal of the project is to assist medical professionals by providing an AI-based decision support system that can classify different stages of Alzheimer’s disease accurately and efficiently.


**Project Overview**

Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory, cognition, and daily functioning. Early diagnosis plays a critical role in slowing disease progression and improving patient quality of life. However, traditional diagnostic methods involving MRI or PET scans require expert interpretation, are time-consuming, costly, and not always accessible.

This project proposes a deep learning–based image classification system that analyzes brain MRI scans and classifies them into four stages of Alzheimer’s disease. The system uses a Convolutional Neural Network (CNN) and includes explainability through Grad-CAM visualizations to improve trust and interpretability.

**Objectives**

To study Alzheimer’s disease progression using brain MRI scans

To build a CNN model for multi-class classification of Alzheimer’s stages

To classify MRI images into:

Non-Demented

Very Mild Dementia

Mild Dementia

Moderate Dementia

To apply image preprocessing techniques for consistency and accuracy

To evaluate the model using accuracy, precision, recall, and confusion matrix

To demonstrate the feasibility of AI-assisted early diagnosis

To provide explainable predictions using Grad-CAM

To build a basic user interface for image upload and prediction

**Key Features**

MRI-based Alzheimer’s stage classification

CNN-based deep learning model

Image preprocessing including resizing and normalization

Grad-CAM based visual explanation of predictions

Four-class disease classification

Local web interface using Flask

Designed as a clinical decision support tool (not a replacement for doctors)

**Technologies Used**
Programming & Frameworks

Python 3.11

TensorFlow with Keras

Flask (for local deployment)

Image Processing & Visualization

Pillow (PIL)

Grad-CAM for explainability

Development Tools

Jupyter Notebook

Visual Studio Code

Git & GitHub for version control

**Dataset**

The dataset consists of labeled brain MRI images categorized into four classes representing different stages of Alzheimer’s disease. Each image represents a brain slice used for supervised learning.

**Methodology**

Data Collection and Understanding
MRI images were collected and analyzed to understand class distribution and disease characteristics.

Image Preprocessing

Grayscale conversion

Image resizing to uniform dimensions

Normalization of pixel values

Channel duplication where required

Model Design and Training

CNN architecture with convolution, pooling, and dropout layers

Adam optimizer and categorical cross-entropy loss

Class weighting and early stopping to prevent overfitting

Model Evaluation

Accuracy

Confusion matrix

Precision and recall

Explainability

Grad-CAM heatmaps to visualize important brain regions influencing predictions

Deployment

Flask-based local web application for image upload and prediction

Results

Achieved high classification accuracy (around 97 percent in local testing)

Clear separation between dementia stages

Grad-CAM visualizations aligned with neurologically relevant brain regions

Demonstrated feasibility of AI-based early Alzheimer’s detection

**Limitations**

Performance may drop on low-quality or noisy MRI images

Cloud deployment faced technical challenges

Requires larger and more diverse datasets for real-world use

**Future Scope**

Training on larger and more diverse datasets

Integration of multimodal data (MRI, PET, cognitive scores, genetics)

Use of advanced architectures such as ResNet, DenseNet, EfficientNet

Full cloud deployment on AWS, Azure, or Google Cloud

Mobile or edge-device deployment for rural and remote healthcare

Secure authentication and data handling


License

This project is developed strictly for academic and educational purposes.
