# Plant Disease Predictor using CNN

## 🌿 Project Overview
This project is a **deep learning-based Plant Disease Predictor** that utilizes **Convolutional Neural Networks (CNNs)** to classify plant leaf images into healthy or diseased categories. The model is trained on the **PlantVillage dataset** and deployed as a **Streamlit web application**, containerized using **Docker** for easy deployment.

## 🛠 Workflow
1. **Image Data Collection** – Using the PlantVillage dataset for training.
2. **Data Processing** – Image augmentation and preprocessing.
3. **Train-Test Split** – Splitting dataset for model training and evaluation.
4. **CNN Training** – Implementing a deep learning model using TensorFlow/Keras.
5. **Model Evaluation** – Assessing accuracy, loss, and classification performance.
6. **Streamlit Web App** – Deploying a user-friendly interface for predictions.
7. **Docker Integration** – Containerizing the application for seamless deployment.

## 📂 Dataset
- The dataset used is **PlantVillage**, which consists of labeled plant leaf images.
- Preprocessing includes resizing, normalization, and augmentation for better performance.

## 🏗 Model Architecture
- **Convolutional Neural Network (CNN)** with multiple layers:
  - Convolutional + Pooling layers for feature extraction
  - Fully connected layers for classification
  - Softmax activation for multi-class prediction
- Optimizer: **Adam**
- Loss Function: **Categorical Cross-Entropy**

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```bash
   git clone https://github.com/your-username/plant-disease-predictor.git
   cd plant-disease-predictor
```

### 2️⃣ Install Dependencies
```bash
   pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
   streamlit run app.py
```

## 🐳 Docker Deployment
To run the project using Docker:
```bash
   docker build -t plant-disease-predictor .
   docker run -p 8501:8501 plant-disease-predictor
```

## 🎯 Usage
1. Upload an image of a plant leaf.
2. The model predicts whether the plant is healthy or diseased.
3. The result is displayed in the Streamlit web interface.

## 📊 Results
- The model achieves **high accuracy** on the test dataset.
- Performance metrics such as **precision, recall, and F1-score** are analyzed.
- Works well for various plant species in the dataset.


Trained model link-https://drive.google.com/file/d/1-IQkH4rruNOS1OhDkFIecEkTiadYZaZG/view?usp=drive_link





