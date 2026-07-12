A deep learning-based Sign Language Recognition system trained on the Word-Level American Sign Language (WLASL) dataset. This repository contains the complete pipeline: data preprocessing, model training, and a real-time deployment interface built using Streamlit.

## 🚀 Features
* **Custom Data Pipeline:** Structured preprocessing that maps, filters, and flattens WLASL JSON annotations against raw `.mp4` video assets.
* **Robust Data Validation:** Automatic detection and handling of missing video frames or broken file references to ensure 100% data integrity before training.
* **Optimized Classification:** Target vocabulary mapped dynamically to a compressed label space for efficient multi-class training.
* **Interactive UI:** A lightweight Streamlit web application that serves the model for inference, allowing users to evaluate sign video predictions instantly.

## 🛠️ Tech Stack
* **Core:** Python
* **Deep Learning Framework:** PyTorch / TensorFlow *(Select yours)*
* **Frontend/Deployment:** Streamlit
* **Data Management:** JSON, OpenCV, OS File Systems
