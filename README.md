# Hybrid Wind Power Forecasting (Wavelet Transform & ML) 

This repository contains my **Graduation Project**, which focuses on high-precision wind power forecasting by combining **Signal Processing (Wavelet Transform)** with **Machine Learning** models. 

##  Project Overview
Wind speed data is inherently "noisy" and non-stationary. This project implements a **Hybrid Approach**:
1. **Wavelet Decomposition:** Using Discrete Wavelet Transform (DWT) to decompose the wind signals into sub-bands and remove noise.
2. **Predictive Modeling:** Applying various ML algorithms on the "denoised" data to predict future energy output.

##  Hybrid Methodology & Models
The core of this project is the comparison of multiple architectures after the Hybrid Wavelet preprocessing phase:

* **Wavelet Transform (DWT):** Used for multi-resolution analysis to capture both high-frequency and low-frequency components of wind data.
* **Random Forest (RF) & Decision Trees (DT):** Ensemble and tree-based methods used to capture non-linear patterns in the denoised signal.
* **Artificial Neural Networks (ANN):** A multi-layer perceptron model optimized to forecast energy output with high accuracy.



##  Key Features
* **Denoising Pipeline:** Significant reduction in Mean Absolute Error (MAE) through Wavelet-based preprocessing.
* **Algorithm Comparison:** Comprehensive analysis and performance metrics (R², MSE, RMSE) for **ANN, RF, and DT**.
* **Time-Series Optimization:** Tailored for short-to-medium term wind power forecasting.

##  Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

* **Language:** Python
* **Libraries:** PyWavelets (for DWT), Scikit-Learn, TensorFlow/Keras, NumPy, Matplotlib.

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/meren01/WindForecasting-HybridWavelet.git](https://github.com/meren01/WindForecasting-HybridWavelet.git)
**Install requirements:**
 ```bash
pip install -r requirements.txt
 
 ```
Developed by Murat Eren Furfuru

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/meren01/WindForecasting-HybridWavelet.git](https://github.com/meren01/WindForecasting-HybridWavelet.git)
