# Rock vs Mine Prediction

## Overview
This machine learning project predicts whether an object detected by a sonar signal is a **rock** or a **mine**. The model is trained on the **Sonar Dataset**, which contains sonar signal data for objects that are either rocks (denoted by "R") or mines (denoted by "M").

The project includes:
- Data preprocessing
- Model training using Logistic Regression
- Evaluation of the model’s performance
- A simple web app for predictions using **Streamlit**.

## Requirements

### Libraries
To run the project, you need to install the following libraries:

- **pandas**: For data manipulation
- **scikit-learn**: For machine learning algorithms and evaluation metrics
- **joblib**: For saving and loading models
- **streamlit**: For building the web application
- **pyngrok**: For exposing the web app over the internet

You can install these libraries using pip:
```bash
pip install pandas scikit-learn joblib streamlit pyngrok
```

## Project Setup

1. **Clone the repository** (if applicable) or create a new notebook or directory to run the code.
2. **Train the model**:
   - The model is trained using the Sonar Dataset. The dataset is loaded, preprocessed, and then split into training and testing sets.
   - We use **Logistic Regression** for binary classification to predict whether the object is a rock or a mine.
   - The trained model is saved as `rock_vs_mine_model.pkl`.

3. **Create the Streamlit Web App**:
   - The app allows users to upload sonar data in CSV format and get a prediction of "Rock" or "Mine".
   - The model is loaded from the saved `.pkl` file, and predictions are made based on the uploaded data.

4. **Run the Web App**:
   - The app is exposed through **ngrok** so you can access it via a public URL.

### Running the App

After setting up your project and training the model, run the following code to start the Streamlit app in Colab:

```python
from pyngrok import ngrok

# Run the app in the background
!streamlit run app.py &

# Expose the app using ngrok
url = ngrok.connect(port='8501')
print(f"App is live at: {url}")
```

Once the app is running, you can open the public URL in your browser and start using it.

## Web App Usage

- **Step 1**: Upload a sonar dataset in CSV format.
- **Step 2**: Press the "Predict" button.
- **Step 3**: Get predictions on whether the object is a "Rock" or "Mine".

## File Structure

```
- app.py                  # Streamlit web app code
- rock_vs_mine_model.pkl  # Trained model file
- requirements.txt        # List of required Python libraries
- README.md               # Project documentation
```

## Model Evaluation

The model is evaluated based on **accuracy** (percentage of correct predictions). You can improve the model by experimenting with different algorithms (e.g., Random Forest, Support Vector Machines) or by tuning hyperparameters.

### Example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/sonar/sonar.all-data"
df = pd.read_csv(url, header=None)

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
