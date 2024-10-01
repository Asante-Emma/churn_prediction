import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

def preprocess_data(input_path, output_path):
    # Load data into pandas dataframe
    df = pd.read_csv(input_path)

    # List of binary categorical columns
    binary_cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

    # List of other categorical columns
    other_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    # List of numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Covert binary categorical columns to numerical
    df[binary_cat_cols] = df[binary_cat_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}))

    # One-hot encode the rest of the categorical columns
    df = pd.get_dummies(df, columns=other_cat_cols, drop_first=True)

    # Replace empty strings with the minimum value of TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
    df['TotalCharges'].fillna(df['TotalCharges'].min(), inplace=True)

    # Feature Scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Drop customerID column
    df.drop(columns=['customerID'], inplace=True)

    # Extract the majority and the minority class
    df_majority = df[df['Churn'] == 0]
    df_minority = df[df['Churn'] == 1]

    # Upsample the minority class
    df_minority_upsampled = resample(df_minority,
                                    replace=True,
                                    n_samples=len(df_majority),
                                    random_state=42)

    # Concatenate the majority class and the upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Drop customerID column
    df_upsampled.drop(columns=['customerID'])
    
    df_upsampled.to_csv(output_path, index=False)