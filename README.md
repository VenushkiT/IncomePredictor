# Income Level Prediction Using Machine Learning

## Project Overview

This project applies various machine learning techniques to predict whether an individual earns more than $50K per year based on the **Adult Income Dataset**. The dataset comprises demographic and employment-related attributes, such as age, education, marital status, and occupation, providing a broad perspective on factors influencing income levels.

The project involves data preprocessing, handling class imbalance, and comparing the performance of multiple machine learning models. Metrics like accuracy, precision, recall, and F1-score are used for evaluation. Additionally, ensemble methods and hyperparameter tuning were implemented to optimize model performance.

## Key Features

### 1. **Data Preprocessing**
- Handled missing values and duplicates.
- Addressed outliers using IQR-based capping.
- Performed feature scaling, normalization, and encoding.
- Applied SMOTE to balance classes and enhance minority class representation.

### 2. **Model Training and Evaluation**
- Algorithms Used:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Decision Tree
  - Random Forest
  - Bernoulli Naive Bayes
  - K-Nearest Neighbors (KNN)
- Ensemble Methods:
  - Voting Classifier combining Random Forest and XGBoost.
- Hyperparameter tuning with Grid Search for optimal model configuration.
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

### 3. **Visualizations**
- Data insights using boxplots, histograms, and heatmaps.
- Model evaluation with confusion matrices, ROC curves, and precision-recall plots.

## Results and Conclusions

### **Logistic Regression**
- Best Configuration: **C=1**
- Accuracy: 70%
- Macro-average recall: 72%
- Demonstrates a good balance between accuracy and recall. Higher C values caused overfitting, while lower values led to underfitting.

### **Support Vector Machines (SVM)**
- **RBF Kernel** outperformed the Linear Kernel:
  - Accuracy: **75.98%**
  - Macro-average recall: **76%**
  - Best Parameters (manually set): **C=200, γ=0.5**
  - Grid Search Parameters: **C=100, γ=1** resulted in slightly lower accuracy (75.26%).

### **Decision Tree Classifier**
- Accuracy: **76%**
- Recall: **76% for <=50K, 77% for >50K**
- Demonstrates strong identification of both classes after hyperparameter tuning.

### **Random Forest and XGBoost Ensemble**
- Accuracy: **77.7%** (hard voting), **77.1%** (soft voting)
- Recall: **78% for <=50K, 76% for >50K**
- Ensemble learning marginally improved recall and accuracy.

### **Bernoulli Naive Bayes**
- Accuracy: **71%**
- Recall: **70% for <=50K, 76% for >50K**
- Precision for >50K: **44%**
- Shows good recall for the minority class but weaker precision for >50K.

### **K-Nearest Neighbors (KNN)**
- Initial Accuracy: **80%**
- Best Configuration (n_neighbors=15): Accuracy improved to **81%**
- Recall: **86% for <=50K, 64% for >50K**
- Best Performing Model:
  - Highest accuracy among all models.
  - Balanced performance with a macro-average recall of **75%**.

### Conclusion
While multiple models delivered competitive results, **KNN emerged as the best-performing model** due to its superior accuracy (81%) and balanced recall. It effectively captured both majority and minority classes, making it the most suitable choice for this classification task.

## Future Work
- Explore additional ensemble methods like Gradient Boosting or LightGBM.
- Fine-tune other hyperparameters for further performance enhancement.
- Deploy the best-performing model as a web service for real-time predictions.
