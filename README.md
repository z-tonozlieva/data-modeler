# DataModeler

## Overview

The `DataModeler` implementation is designed to simplify the process of preparing data, training a machine learning model, and making predictions. 
It can handle transactional data with features like transaction amount, date, and a binary outcome.

## Features

- Data preparation and cleaning
- Handling of missing values through imputation
- Feature engineering from date fields
- Model training (currently using Logistic Regression)
- Prediction on new data
- Save and load functionality
- Model summary generation

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Installation

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Initializing the DataModeler

```python
import pandas as pd
from data_modeler import DataModeler

# Prepare your data
data = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01',
            '2022-08-01',
            None,
            '2022-12-01',
            '2022-02-01',
            None,
            '2022-02-01',
            '2022-01-01',
            '2022-11-01',
            '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)

# Create an instance of DataModeler
modeler = DataModeler(data)
```

### Fitting the Model

```python
modeler.fit()
```

### Getting Model Summary

```python
summary = modeler.model_summary()
print(summary)
```

### Making Predictions

```python
# On training data
predictions = modeler.predict()

# On new data
new_data = pd.DataFrame({
    "customer_id": [16, 17],
    "amount": [2, 8],
    "transaction_date": ['2022-03-01', '2022-07-01'],
})
new_predictions = modeler.predict(new_data)
```

### Saving and Loading the Model

```python
# Save the model
modeler.save('transact_modeler.pkl')

# Load the model
loaded_modeler = DataModeler.load('transact_modeler.pkl')
```

## Class Methods

- `__init__(self, sample_df: pd.DataFrame)`: Initialize the DataModeler with sample data.
- `prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame`: Prepare data for modeling.
- `impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame`: Impute missing values.
- `fit(self) -> None`: Fit the model on the training data.
- `model_summary(self) -> str`: Generate a summary of the fitted model.
- `predict(self, oos_df: pd.DataFrame = None) -> pd.Series`: Make predictions using the fitted model.
- `save(self, path: str) -> None`: Save the model to a file.
- `load(path: str) -> DataModeler`: Load a saved model from a file.

## Customization

The current implementation uses Logistic Regression as the underlying model. You can easily extend the class to use different models by modifying the `__init__` and `fit` methods.

## Note

This implementation assumes that the 'outcome' column is the target variable. If your target variable has a different name, you'll need to modify the `__init__` method accordingly.