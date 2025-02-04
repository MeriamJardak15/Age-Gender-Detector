# 🎭 AI-Powered Gender and Age Detection Using CNNs  

## 📌 Overview  
This project explores the applications of **Artificial Intelligence (AI)** in **gender and age detection** using **Convolutional Neural Networks (CNNs)**. By addressing key challenges in facial analysis, the study leverages deep learning to enhance prediction accuracy and performance.  

## 📊 Dataset: Adience  
The **Adience** dataset, consisting of **12,000 diverse facial images**, serves as the foundation for this study. The dataset categorizes:  
✅ **Age into 8 groups** (labeled from 0 to 7).  
✅ **Gender as binary values** (0 for male, 1 for female).  

## 🛠️ 3.1 Data Preparation  
To ensure structured and high-quality data input for our CNN models, we performed rigorous data preprocessing:  

1️⃣ **Structured Dataframes**:  
   - Created two separate dataframes:  
     - **df_age**: Contains age categories for each individual.  
     - **df_gender**: Stores gender labels.  

2️⃣ **Data Extraction**:  
   - Extracted key identifiers from Adience’s text files:  
     ✅ **Directory path**  
     ✅ **Image reference**  
     ✅ **Face ID** (unique identifier)  
     ✅ **Age category** (0-7)  
     ✅ **Gender label** (0: Male, 1: Female)  

3️⃣ **Global Dataframe Consolidation**:  
   - Merged `df_age` and `df_gender` into a **single structured dataframe (`df_total`)** using `face_id` as the primary key.  
   - This ensures that all essential information—**image paths, age, and gender labels**—are stored in one place for efficient model training.  

---

## 🏗️ Methodology  
We implemented two distinct CNN-based approaches to compare their effectiveness:  

1️⃣ **Separate CNN Models** for age and gender detection.  
2️⃣ **Unified CNN Model** for simultaneous detection.  

Each approach underwent rigorous **training and testing phases** to ensure robust evaluation.  

## ⚙️ Tools & Libraries  
The implementation relies on **key libraries** for data processing, model creation, and evaluation:  

🔹 **Data Processing & Manipulation**: `numpy`, `pandas`  
🔹 **Image Processing**: `cv2`  
🔹 **Data Visualization**: `matplotlib`, `seaborn`  
🔹 **Deep Learning Framework**: `tensorflow`, `keras`  
🔹 **Model Layers**: `Dense`, `Conv2D`, `Flatten`  
🔹 **Optimization**: `Adam`  
🔹 **Label Encoding**: `scikit-learn` (`LabelEncoder`, `to_categorical`)  

## 📈 Results  
Our findings demonstrate the power of CNNs in gender and age classification:  

✅ **92% Accuracy** – Separate CNN models  
✅ **77% Accuracy** – Unified CNN model  

These results underscore the effectiveness of CNN-based architectures, with further research needed to optimize model parameters and achieve **100% accuracy**.  

## 🚀 Future Improvements  
To push the boundaries of AI-driven facial analysis, we aim to:  

📌 **Optimize CNN architectures** for higher accuracy.  
📌 **Fine-tune hyperparameters** to improve learning efficiency.  
📌 **Experiment with transfer learning** using pre-trained models like VGG16 or ResNet.  
📌 **Expand dataset diversity** for more generalized predictions.  


