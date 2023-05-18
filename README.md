## Code Documentation: Social Network Ads Classification

This code is for building a classification model to predict whether a user will purchase a product based on social network ads. It uses various machine learning algorithms and evaluates their performance.

### 1. Importing Libraries
The necessary libraries are imported in the code to perform various tasks related to data manipulation, visualization, model training, and evaluation. The libraries used are:
- `pandas`: For data handling and manipulation.
- `matplotlib.pyplot`: For data visualization.
- `sklearn` (Scikit-learn): For machine learning algorithms and evaluation metrics.

### 2. Loading the Dataset
The code reads the dataset from a CSV file (`Social_Network_Ads.csv`) using the `read_csv` function from `pandas`. The dataset contains information about social network ads and whether a user purchased the product or not.

### 3. Exploratory Data Analysis
Several exploratory data analysis tasks are performed to understand the dataset:

- Printing the shape of the dataset using `dataset.shape` to display the number of rows and columns.
- Displaying the first 20 rows of the dataset using `dataset.head(20)`.
- Generating descriptive statistics of the dataset using `dataset.describe()`, which includes count, mean, standard deviation, minimum, and quartile values for each column.
- Printing the class distribution using `dataset.groupby('Purchased').size()` to show the number of instances in each class.

### 4. Data Preprocessing
The dataset is preprocessed to prepare it for model training:

- The features and target variable are extracted from the dataset. The features (`x`) are selected from columns 2 and 3, and the target variable (`y`) is selected from column 4.
- The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`. The testing set will contain 30% of the data, while the training set will contain the remaining 70%.

### 5. Model Evaluation
Several classification algorithms are evaluated using cross-validation:

- A list `models` is created to store the algorithms to be evaluated. The algorithms include Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Tree Classifier, Gaussian Naive Bayes, and Support Vector Machine.
- The code then performs a cross-validation loop for each algorithm using `StratifiedKFold` with 10 splits. It evaluates the accuracy of each algorithm on the training set using `cross_val_score` and stores the results in `results`. The algorithm names are stored in the `names` list.
- For each algorithm, the code prints the mean accuracy and standard deviation of the cross-validated results.

### 6. Model Training and Prediction
The Support Vector Machine (SVM) algorithm is chosen as the final model based on the evaluation results. The code performs the following steps:

- A new SVM model is instantiated with `'auto'` as the gamma value.
- The model is trained on the training set using `fit` method with `x_train` and `y_train`.
- Predictions are made on the testing set using the trained model with `predict` method and stored in `predictions`.

### 7. Model Evaluation and Reporting
The predictions made by the SVM model are evaluated and reported:

- The accuracy of the model is calculated by comparing the predicted values (`predictions`) with the actual values (`y_test`). The accuracy score is multiplied by 100 to represent it as a percentage.
- The accuracy score is printed using `print(str("%.2f" % accuracy)+' %')`.
- The confusion matrix is printed using `confusion_matrix` to show the performance of the model in terms of true positive, true negative, false positive, and false negative predictions.
- The classification report is printed using `classification_report` to display precision, recall, F1-score, and support for each class.
