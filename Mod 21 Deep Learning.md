


## Analysis of Deep Learning Model for Alphabet Soup Charity Success Prediction- Sunil Williams

### Overview

The purpose of this analysis is to develop a binary classification model that can predict whether applicants will use Alphabet Soup's funding successfully. By analyzing various features from the application dataset, the model aims to identify which organizations have the highest probability of success, enabling Alphabet Soup to make more informed decisions and optimize their resource allocation.

### Results

#### Data Preprocessing

**Target Variable:**

-   `IS_SUCCESSFUL`: Binary indicator (1 = successful, 0 = unsuccessful) representing whether the money was used effectively

**Feature Variables:**

-   `APPLICATION_TYPE`: Application category
-   `AFFILIATION`: Affiliated sector of industry
-   `CLASSIFICATION`: Government organization classification
-   `USE_CASE`: Use case for funding
-   `ORGANIZATION`: Organization type
-   `STATUS`: Active status (in first model)
-   `INCOME_AMT`: Income classification
-   `SPECIAL_CONSIDERATIONS`: Special considerations for application (in first model)
-   `ASK_AMT`: Funding amount requested

**Variables Removed:**

-   `EIN`: Employer Identification Number (removed in both models)
-   `NAME`: Organization name (removed in first model, binned in optimization)
-   `STATUS`: Removed in optimization
-   `SPECIAL_CONSIDERATIONS`: Removed in optimization

#### Data Transformation Steps

1.  Binned rare categorical values:
    -   For `APPLICATION_TYPE`, values with counts < 500 were grouped as "Other"
    -   For `CLASSIFICATION`, values with counts < 1000 were grouped as "Other"
    -   In optimization, `NAME` values with counts ≤ 5 were grouped as "Other"
2.  Converted categorical variables to numeric using one-hot encoding
3.  Split the data into training and testing sets
4.  Scaled the features using StandardScaler

#### Compiling, Training, and Evaluating the Model

**Initial Model Architecture:**

-   Input features: 43 (after preprocessing)
-   First hidden layer: 80 neurons with ReLU activation
-   Second hidden layer: 30 neurons with ReLU activation
-   Output layer: 1 neuron with sigmoid activation
-   Loss function: Binary crossentropy
-   Optimizer: Adam
-   Metrics: Accuracy
-   Training: 100 epochs

**Initial Model Performance:**

-   Loss: 0.557
-   Accuracy: 72.94%

**Optimization Attempts:**

1.  **Feature Engineering:**
    -   Removed additional variables (STATUS, SPECIAL_CONSIDERATIONS)
    -   Added NAME as a feature, binning values with frequency ≤ 5 as "Other"
2.  **Model Architecture Changes:**
    -   Increased first hidden layer to 100 neurons
    -   Changed second hidden layer's activation to sigmoid
    -   Added third hidden layer with 10 neurons and sigmoid activation
3.  **Alternative Model:**
    -   Implemented a Random Forest Classifier with 128 estimators as a comparison

**Optimization Model Performance:**

-   Neural Network (Optimized):
    -   Loss: 0.539
    -   Accuracy: 73.36%
-   Random Forest Classifier:
    -   Accuracy: 72.27%

The target model performance of 75% accuracy was not achieved despite the optimization efforts. The optimized neural network model showed a modest improvement of approximately 0.42 percentage points over the initial model.

### Summary

The deep learning models developed for Alphabet Soup achieved moderate success with accuracies between 72-73%. While the optimization efforts improved performance slightly, they did not reach the target 75% accuracy threshold.

The Random Forest model performed similarly to the neural network models, suggesting that the dataset's predictive signals may have limitations that cannot be easily overcome by simply changing model architectures.

#### Recommendation

For this classification problem, I would recommend:

1.  **Ensemble Learning Approach**: Combining multiple models (such as neural networks, random forests, and gradient boosting) through techniques like stacking or voting could potentially improve prediction accuracy.
2.  **Additional Data**: If possible, incorporating more data points or additional features about the organizations could significantly enhance model performance. External data like financial health metrics, leadership information, or historical performance metrics might provide stronger signals for success prediction.

This recommendation is based on the observation that both the neural network and random forest models reached similar performance levels, suggesting we may need to improve the input data quality rather than just tuning model architecture.
