
# Dhondt-XAI

Dhondt-XAI is a Python library that applies the D'Hondt method to feature importances of decision tree models.
This allows for a proportional allocation of feature weights in an explainable manner.

## Installation
To install the library, run:
```
pip install dhondt-xai
```

## Usage
This library can be used with tree-based models such as Random Forest, XGBoost, CatBoost, and AdaBoost. You can
configure alliances between features and apply thresholds for feature importance analysis.

## Example
```python
from dhondt_xai.dhondt_xai import DHondtXAI
from dhondt_xai.plot import plot_dhondt_results

# Example usage code goes here
```
