# Music Genre Classification Project

## Overview
In this project, we undertook the challenge of music genre classification through a comprehensive approach that explored various methodologies, ranging from traditional machine learning to state-of-the-art deep learning models. Our work was structured into four main parts, each building upon the findings of the previous phase to develop an increasingly robust classification system.

## Data Processing & Feature Extraction

**Dataset Description:**  

- **Audio Files**: The dataset consists of **1,000 audio tracks**, each lasting **30 seconds**.  

- **Genres**: There are **10 genres**, each represented by **100 tracks**:  
  - Blues  
  - Classical  
  - Country  
  - Disco  
  - Hip Hop  
  - Jazz  
  - Metal  
  - Pop  
  - Reggae  
  - Rock
    
Our initial work focused on generating rich representations of audio data from the GTZAN dataset, which provides 30-second audio samples. We implemented an extensive feature extraction process that captured both spectral and temporal characteristics of the audio signals. The primary features we extracted included mel-spectrograms and 20 MFCC coefficients. We enhanced our feature set with additional parameters including Chroma Features, Root Mean Square (RMS) Energy, Spectral Features, Zero Crossing Rate (ZCR), Harmonic and Perceptual Features, and Tempo.

To create a more granular dataset, we implemented two parallel approaches. First, we processed the full 30-second audio segments, extracting all mentioned features. Additionally, we split each 30-second audio into ten 3-second chunks, applying the same feature extraction process to these shorter segments. This dual approach allowed us to investigate the impact of segment length on classification performance.

## Exploratory Data Analysis
Following the feature extraction phase, we conducted a comprehensive exploratory data analysis (EDA) on the 30-second and the 3-second segments. This analysis provided valuable insights into the distribution and characteristics of our extracted features across different genres. While we maintained detailed internal documentation of our EDA findings, we chose to keep this section concise in our public documentation (README).

![output](https://github.com/user-attachments/assets/16064baf-9a66-4638-a3ba-07f9c6f95630)

![audio_wave](https://github.com/user-attachments/assets/df35566e-f184-4e03-8b66-24de43bddff3)


## Model Development

### Traditional Machine Learning Approaches
Our initial modeling phase focused on evaluating various shallow models on both our 30-second and 3-second datasets. We implemented and tuned several algorithms, including Support Vector Machines (SVM), Logistic Regression, Random Forests, Decision Trees, XGBoost, and KNeighborsClassifier. For each model, we used both Bayesian optimization and grid search for hyperparameter fine-tuning, selecting the most suitable approach based on the scenario and we carefully tuned the train_test_split function parameters to ensure robust evaluation.

The results revealed an interesting pattern: models trained on 3-second segments consistently outperformed those trained on 30-second segments. This improvement was attributed to the larger dataset size and more detailed feature representation available in the shorter segments. Among the models trained on 3-second data, the KNeighborsClassifier emerged as the top performer. For the 30-second segments, SVM showed superior performance, which aligns with its known efficiency on smaller datasets.

However, we encountered a significant limitation with these shallow models. While they performed well on the test set drawn from the GTZAN dataset, they struggled to generalize effectively to music samples outside this dataset. This limitation prompted us to explore more sophisticated approaches.

### Deep Learning Approach
To address the generalization challenges observed with shallow models, we developed a Convolutional Neural Network (CNN) architecture specifically designed for processing mel-spectrogram data from 30-second segments. Our CNN implementation followed a VGG-like structure, incorporating multiple convolutional layers for feature extraction, followed by max pooling layers to reduce spatial dimensions and improve computational efficiency. We integrated dropout layers throughout the network to combat overfitting and enhance generalization capabilities.

The model's architecture progressed from convolutional layers through a flattening operation before connecting to fully connected layers, ultimately terminating in a softmax layer for genre probability distribution. We employed K-Fold cross-validation during training to ensure robust performance and minimize overfitting risks.

The CNN demonstrated significantly improved generalization capabilities compared to our shallow models, performing well both on the test set and on music samples outside the GTZAN dataset. However, this improved performance came at a cost - the final model size reached 554MB, presenting potential deployment challenges.

### DistilHuBERT Implementation
In our final approach, we explored the use of DistilHuBERT, a distilled version of the HuBERT model, to achieve a better balance between model performance and size. We utilized the model through a feature extraction pipeline that included automatic feature extraction with normalization and attention mask generation. The implementation involved setting up a classification head atop the DistilHuBERT base model, configured with appropriate label mappings for our genre classification task.

This approach proved particularly successful, achieving slightly better generalization than our CNN model while dramatically reducing the model size to just 94MB - less than one-fifth of the CNN model's size. The significant reduction in model size, combined with maintained or improved performance, makes this our most promising approach for practical applications.

### Model Deployment and Interactive Interface
To make our work accessible and practical for real-world use, we developed an interactive interface that showcases our genre classification system. The interface provides a simple and intuitive environment where users can upload audio files and receive genre predictions in real-time. We chose to deploy the DistilHuBERT model due to its optimal balance of performance and size, making it well-suited for deployment in resource-constrained environments.

The system handles audio processing and prediction generation, providing quick and accurate results. You can access and try our application at: [Music Genre Classification - Yourkln](https://yourkln.com/mcproject)

## Conclusion
Through our systematic exploration of various approaches to music genre classification, we've demonstrated the trade-offs between model complexity, performance, and practicality. While our shallow models provided good performance on in-distribution data, their limited generalization capabilities led us to explore deep learning approaches. The CNN model addressed these generalization issues but at the cost of model size. Our final DistilHuBERT implementation represents an optimal balance, providing robust generalization capabilities in a significantly more compact model. This progression illustrates the importance of considering both technical performance and practical constraints in developing real-world machine learning solutions.
