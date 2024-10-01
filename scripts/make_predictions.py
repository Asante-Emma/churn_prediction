import joblib
import pandas as pd


class PredictChurn():
    def __init__(self, input_data):
        self.input_data = input_data
        # Load model and scaler
        self.model = joblib.load('../models/best_churn_model.joblib')
        self.scaler = joblib.load('../models/scaler.joblib')

    def _preprocess_input(self):
        """
        Preprocess the input data by scaling specific columns.
        
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.
        """
        # Ensure the columns are in the correct order
        columns = ['tenure', 'TotalCharges', 'MonthlyCharges'] + \
                  [col for col in self.input_data.columns if col not in ['tenure', 'TotalCharges', 'MonthlyCharges']]
        data = self.input_data[columns]

        # Apply scaling to the specified columns
        data[['tenure', 'MonthlyCharges', 'TotalCharges']] = self.scaler.transform(
            data[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )
        
        return data

    @property
    def predict_churn(self):
        """
        Predict churn based on the preprocessed input data.
        
        Returns:
            np.ndarray: Array of predictions (0 or 1) for churn.
        """
        # Preprocess the input data
        processed_data = self._preprocess_input()

        # Make predictions
        predictions = self.model.predict(processed_data)
        
        return predictions
    
if __name__ == '__main__':
    new_data = pd.DataFrame({
        'tenure': [9, 54, 7],
        'TotalCharges': [45, 81, 42],
        'MonthlyCharges': [3831, 2320, 2957],
        'Contract_Two year': [False, True, False],
        'InternetService_Fiber optic': [True, False, True],
        'PaymentMethod_Electronic check': [True, False, False],
        'Contract_One year': [False, False, False],
        'gender': [0, 0, 0],
        'TechSupport_Yes': [False, False, False],
        'OnlineSecurity_Yes': [False, True, False],
        'Partner': [0, 0, 0],
        'PaperlessBilling': [1, 1, 0]
    })


    # Create an instance of PredictChurn
    predictor = PredictChurn(new_data)

    # Get predictions
    predictions = predictor.predict_churn
    print(list(predictions))