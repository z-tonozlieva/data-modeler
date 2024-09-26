import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        '''
        Initialize the DataModeler as necessary.
        '''
        self.sample_df = sample_df.copy()
        self.target_column = 'outcome'
        self.feature_columns = ['amount', 'transaction_date']
        self.model = LogisticRegression()
        self.imputer = SimpleImputer(strategy='mean')

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        '''
        df = oos_df if oos_df is not None else self.sample_df
        
        # Convert transaction_date to datetime and extract relevant features
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_day'] = df['transaction_date'].dt.day
        
        # Select relevant columns
        columns_to_use = ['amount', 'transaction_month', 'transaction_day']
        
        return df[columns_to_use]

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        '''
        df = oos_df if oos_df is not None else self.sample_df
        prepared_df = self.prepare_data(df)
        
        if oos_df is None:
            self.imputer.fit(prepared_df)
        
        imputed_data = self.imputer.transform(prepared_df)
        imputed_df = pd.DataFrame(imputed_data, columns=prepared_df.columns, index=prepared_df.index)
        
        return imputed_df

    def fit(self) -> None:
        '''
        Fit the model of your choice on the training data passed in the constructor,
        assuming it has been prepared by the functions prepare_data and impute_missing
        '''
        X = self.impute_missing()
        y = self.sample_df[self.target_column]
        self.model.fit(X, y)

    def model_summary(self) -> str:
        '''
        Create a short summary of the model you have fit.
        '''
        X = self.impute_missing()
        y = self.sample_df[self.target_column]
        y_pred = self.model.predict(X)
        
        summary = f"Model: Logistic Regression\n"
        summary += f"Number of features: {X.shape[1]}\n"
        summary += f"Number of samples: {X.shape[0]}\n"
        summary += f"Model coefficients:\n{self.model.coef_}\n"
        summary += f"Model intercept: {self.model.intercept_}\n"
        summary += f"Classification Report:\n{classification_report(y, y_pred)}"
        
        return summary

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series:
        '''
        Make a set of predictions with your model.
        Assume the data has been prepared by the functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        '''
        X = self.impute_missing(oos_df)
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index, name='predicted_outcome')

    def save(self, path: str) -> None:
        '''
        Save the DataModeler so it can be re-used.
        '''
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> 'DataModeler':
        '''
        Reload the DataModeler from the saved state so it can be re-used.
        '''
        with open(path, 'rb') as file:
            return pickle.load(file)