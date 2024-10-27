
# Import necessary libraries
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from dhondtxai import DhondtXAI, plot_parliament  # Import the DhondtXAI class and plot_parliament function

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/glioma/TCGA_InfoWithGrade.csv')  # Adjust the path according to your setup

# Separate the features and the target variable
X = data.drop(['Grade'], axis=1)  # Assuming 'Grade' is the target variable
y = data['Grade']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a CatBoost classifier
model = CatBoostClassifier(verbose=0, eval_metric='Accuracy')

# Use DhondtXAI for model training and feature importance analysis
dhondtxai = DhondtXAI(model)

# Train the model using the DhondtXAI class - pass both X_train and y_train
dhondtxai.fit(X_train, y_train)

# Kullanıcı dostu arayüz ile ittifakları ve dışlanacak değişkenleri seçin
alliances, exclude_features = dhondtxai.select_features(X.columns)

# Specify parameters for the D'Hondt method dynamically
num_votes = int(input("Toplam oy sayısını girin: "))
num_mps = int(input("Toplam vekil sayısını girin: "))
threshold_input = input("Eşik değerini (yüzde olarak) girin veya 'None' yazın: ")
threshold = None if threshold_input.lower() == 'none' else float(threshold_input)

# Apply the D'Hondt method
features, votes = dhondtxai.apply_dhondt(
    num_votes=num_votes,
    num_mps=num_mps,
    threshold=threshold,
    alliances=alliances,
    exclude_features=exclude_features
)

# Display the results
print("Votes per Feature/Alliance:")
for feature, vote in zip(features, votes):
    print(f"{feature}: {int(vote)} votes")

# Use the D'Hondt method to allocate seats
seats = dhondtxai.dhondt_method(votes, num_mps)

# Display D'Hondt results
print("\nD'Hondt Method Results:")
for feature, seat in zip(features, seats):
    print(f"{feature}: {seat} MPs")

# Plot the results using plot_parliament
dhondtxai.plot_results(features, seats)

# Additional: Create a visual representation of the parliament using plot_parliament
plot_parliament(
    total_seats=num_mps,
    features=features,
    seats=seats,
    slices=50,  # Customize the number of slices for visualization
    additional_rows=5  # Adjust inner space rows if needed
)
