def ml_model_route():
    print("Stage 1: Exploratory Data Analysis (EDA)")
    nan = input("Are there NaN values in your dataset? (y/n): ")
    if nan.lower() == 'y':
        feature = input("Which feature has NaN values?: ")
        dtype = input(f"Is '{feature}' an integer type? (y/n): ")
        if dtype.lower() == 'y':
            print(f"Replace NaN in '{feature}' with the mode value.")
        else:
            print(f"Consider replacing NaN in '{feature}' with a suitable value (mean, median, or other).")
        done = input("Is NaN handling done? (y/n): ")
        if done.lower() != 'y':
            print("Please finish handling NaN values before proceeding.")
            return

    print("\nStage 2: Data Preprocessing")
    datetime_vars = input("Are variables for date and time switched to a DateTime variable? (y/n): ")
    if datetime_vars.lower() != 'y':
        print("Convert date and time variables to pandas DateTime format.")

    scaled = input("Is the data MinMax scaled? (y/n): ")
    if scaled.lower() != 'y':
        print("Consider scaling your data using MinMaxScaler or StandardScaler.")

    categorical = input("Are there any categorical variables? (y/n): ")
    if categorical.lower() == 'y':
        print("Make sure to encode categorical variables using one-hot encoding or label encoding.")

    labeled = input("Is all of the data labeled? (y/n): ")
    if labeled.lower() != 'y':
        print("This is an unsupervised ML problem.")
        manual_label = input("Do you have to manually label this data? (y/n): ")
        if manual_label.lower() == 'y':
            print("Consider manual annotation or semi-supervised approaches.")
        else:
            print("Possible unsupervised ML models: K-Means Clustering, DBSCAN, Hierarchical Clustering, PCA, Autoencoders.")
        return

    print("\nStage 3: Data Modeling")
    standardized = input("Is the data standardized and encoded? (y/n): ")
    if standardized.lower() != 'y':
        print("Ensure data is standardized and categorical variables are encoded before modeling.")
        return

    target_type = input("Are you predicting a numerical or categorical variable? (numerical/categorical): ")
    if target_type.lower() == 'numerical':
        print("Recommended ML models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, SVR.")
    elif target_type.lower() == 'categorical':
        print("Recommended ML models: Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, SVM, KNN Classifier.")
    else:
        print("Please specify 'numerical' or 'categorical'.")

if __name__ == "__main__":
    ml_model_route()