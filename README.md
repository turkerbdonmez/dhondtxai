
# DhondtXAI Library - v12b

The `DhondtXAI` library provides explainability tools for decision tree models, offering methods for feature importance analysis using the D'Hondt method, and visualization tools like bar charts and parliament-style plots.

## Features
- **D'Hondt Method for Feature Importance:** Proportionally allocate votes and seats to features.
- **Bar Plot Visualization:** Display positive and negative feature correlations.
- **Parliament Plot Visualization:** Visualize feature allocations in a parliament-style chart.

---

## Installation
To install the library, use the following command:
```bash
pip install dhondtxai
```

---

## Usage Examples

### 1. D'Hondt Method with Decision Tree Models

This example shows how to use the `DhondtXAI` class with a CatBoost model:

```python
from dhondtxai import DhondtXAI, plot_parliament

# Initialize and train the model
dhondtxai = DhondtXAI(model)
dhondtxai.fit(X_train, y_train)

# Select feature alliances and exclusions
alliances, exclude_features = dhondtxai.select_features(X.columns)

# Apply the D'Hondt method and allocate seats
features, votes = dhondtxai.apply_dhondt(num_votes, num_mps, threshold, alliances, exclude_features)
seats = dhondtxai.dhondt_method(votes, num_mps)

# Plot the results
dhondtxai.plot_results(features, seats)
plot_parliament(total_seats=num_mps, features=features, seats=seats)
```

### 2. Bar Plot Visualization

Displays feature correlations:
- **Blue:** Positive correlation
- **Red:** Negative correlation

```python
dhondtxai.plot_results(features, seats)
```

### 3. Parliament Plot Visualization

Visually represents seat allocations in a semi-circular layout:

```python
plot_parliament(total_seats=600, features=features, seats=seats, slices=50, additional_rows=5)
```

#### Input Parameters:
1. **`num_votes` (int):** Total number of votes to distribute.
2. **`num_mps` (int):** Total number of seats.
3. **`threshold` (float or None):** Optional, to filter low-impact features.

---

## Notes
- Ensure your dataset is preprocessed (e.g., handle missing values, encode categories) before using the library.
- Compatible with decision tree models like `RandomForest`, `XGBoost`, `CatBoost`, and `AdaBoost`.
