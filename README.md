
# DhondtXAI Library - v12

The `DhondtXAI` library is designed to provide advanced explainability tools for machine learning models, particularly decision tree-based models. This library includes methods for applying the D'Hondt method for feature importance analysis, plotting results as bar charts and parliament-style graphs, and integrating seamlessly with various models like CatBoost, XGBoost, Random Forest, and more.

## Features
- **D'Hondt Method for Feature Importance:** Allocate votes and seats for features or alliances using the D'Hondt proportional representation method.
- **Bar Plot Visualization:** Display feature correlations, highlighting positive correlations in blue and negative correlations in red.
- **Parliament Plot Visualization:** Create a parliament-style graph to visualize feature allocations based on the D'Hondt method.

---

## Installation
To install the library, use the following command:
```bash
pip install dhondt_xai-0.2-py3-none-any.whl
```

---

## Usage

### 1. D'Hondt Method with Decision Tree Models

The following example demonstrates how to use the `DhondtXAI` class with a CatBoost model:

```python
# Import necessary libraries
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from dhondt_xai import DhondtXAI, plot_parliament

# Load and prepare the dataset
data = pd.read_csv('/path/to/your/dataset.csv')
X = data.drop(['Target'], axis=1)
y = data['Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model using DhondtXAI
model = CatBoostClassifier(verbose=0, eval_metric='Accuracy')
dhondt_xai = DhondtXAI(model)
dhondt_xai.fit(X_train, y_train)

# User selects feature alliances and exclusions
alliances, exclude_features = dhondt_xai.select_features(X.columns)

# Dynamically enter parameters for the D'Hondt method
num_votes = int(input("Enter the total number of votes: "))
num_mps = int(input("Enter the total number of seats: "))
threshold_input = input("Enter the threshold (percentage) or 'None': ")
threshold = None if threshold_input.lower() == 'none' else float(threshold_input)

# Apply the D'Hondt method
features, votes = dhondt_xai.apply_dhondt(num_votes, num_mps, threshold, alliances, exclude_features)

# Allocate seats using the D'Hondt method
seats = dhondt_xai.dhondt_method(votes, num_mps)

# Plot results
dhondt_xai.plot_results(features, seats)
plot_parliament(total_seats=num_mps, features=features, seats=seats)
```

### 2. Bar Plot Visualization

The bar plot function helps to visualize the correlation between features and the target variable:
- **Positive correlations** are shown in **blue**.
- **Negative correlations** are shown in **red**.

Bar plots are automatically generated when `plot_results` is called. The heights of the bars represent the correlation strength, and the colors signify the direction of the correlation.

**Example:**
```python
dhondt_xai.plot_results(features, seats)
```

### 3. Parliament Plot Visualization

The parliament plot visually represents the allocation of seats across different features or alliances using a half-circle (semi-circular) layout. The visual output is designed to mimic a parliamentary seating arrangement, which makes it intuitive to understand the feature distribution.

#### Input Parameters:
- **`total_seats` (int):** The total number of seats to be allocated.
- **`features` (list of str):** Names of features or alliances.
- **`seats` (list of int):** Number of seats allocated to each feature.
- **`slices` (int):** Number of slices in the semi-circle. Adjust this to control the granularity of the seating arrangement. Higher values provide more detail.
- **`additional_rows` (int):** Number of inner rows to create space. Use this to manage the appearance of the inner circle in the plot.

**Example:**
```python
plot_parliament(
    total_seats=600,
    features=["Feature A", "Feature B", "Feature C"],
    seats=[200, 150, 250],
    slices=50,
    additional_rows=5
)
```

#### Explanation of Hyperparameters:
- **`slices`:** Determines how many segments the parliament plot will have. Increasing this value will result in a finer distribution, which is useful for showing subtle differences in seat allocation.
- **`additional_rows`:** Adds spacing between the inner part of the parliament plot and the outer seats. This can help distinguish between different groups or layers.

---

## Input Parameters and Their Purpose:
1. **`num_votes` (int):** Represents the total number of votes or importance weight that will be distributed among features or alliances. This is critical for proportionally allocating resources or priorities.
2. **`num_mps` (int):** Total number of seats or representatives. Defines how granular the feature representation will be.
3. **`threshold` (float or None):** Optional threshold to exclude features below a certain percentage. Use this to filter out features with lower influence.

---

## Notes
- Make sure your dataset is preprocessed properly (e.g., handling missing values, encoding categorical features) before applying the `DhondtXAI` functions.
- The library integrates seamlessly with common decision tree models such as `RandomForest`, `XGBoost`, `CatBoost`, and `AdaBoost`.

For further details, please refer to the documentation or examples provided in the `examples/` directory.
