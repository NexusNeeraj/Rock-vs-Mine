# Rock vs Mine Prediction

## Overview
This machine learning project predicts whether an object detected by a sonar signal is a **rock** or a **mine**. The model is trained on the **Sonar Dataset**, which contains sonar signal data for objects that are either rocks (denoted by "R") or mines (denoted by "M").

The project includes:
- Data preprocessing
- Model training using Logistic Regression
- Evaluation of the model’s performance

## Requirements

### Libraries
To run the project, you need to install the following libraries:

- **pandas**: For data manipulation
- **scikit-learn**: For machine learning algorithms and evaluation metrics

You can install these libraries using pip:
```bash
pip install pandas scikit-learn
```

## Project Setup

1. **Clone the repository** (if applicable) or create a new notebook or directory to run the code.
2. **Train the model**:
   - The model is trained using the Sonar Dataset. The dataset is loaded, preprocessed, and then split into training and testing sets.
   - We use **Logistic Regression** for binary classification to predict whether the object is a rock or a mine.
   - The trained model is saved as `rock_vs_mine_model.pkl`.





## Model Evaluation

The model is evaluated based on **accuracy** (percentage of correct predictions). You can improve the model by experimenting with different algorithms (e.g., Random Forest, Support Vector Machines) or by tuning hyperparameters.

### Example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
df = pd.read_csv(/content/sonardata.csv, header=None)

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({'R': 0, 'M': 1})  # Encode: Rock = 0, Mine = 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
```

## License

This project is open-source. Feel free to contribute, fork, and share!

---
