
import numpy as np
import matplotlib.pyplot as plt

class DhondtXAI:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.feature_importances = None
    
    def fit(self, X_train):
        # Train the provided model and extract feature importances
        self.model.fit(X_train)
        self.features = X_train.columns
        self.feature_importances = self.model.feature_importances_
    
    def apply_dhondt(self, num_votes, num_mps, threshold=None, alliances=None, exclude_features=None):
        # Remove excluded features from the feature list
        features = [f for f in self.features if f not in exclude_features] if exclude_features else self.features
        
        # Calculate feature importance for alliances
        feature_importances = self.feature_importances[:len(features)]
        grouped_importances = []
        grouped_features = []

        used_features = set()

        if alliances:
            for alliance in alliances:
                total_importance = sum(self.feature_importances[features.index(var.strip())] for var in alliance if var.strip() in features)
                if total_importance > 0:
                    alliance_name = ' + '.join([var.strip() for var in alliance])
                    grouped_features.append(alliance_name)
                    grouped_importances.append(total_importance)
                    used_features.update(alliance)
        
        # Add remaining individual features
        for feature, importance in zip(features, feature_importances):
            if feature not in used_features:
                grouped_features.append(feature)
                grouped_importances.append(importance)
        
        grouped_importances = np.array(grouped_importances)
        
        # Calculate votes per feature/alliance
        votes_per_feature = (grouped_importances / grouped_importances.sum()) * num_votes

        # Apply threshold
        if threshold:
            threshold_votes = (threshold / 100) * num_votes
            votes_per_feature = np.where(votes_per_feature < threshold_votes, 0, votes_per_feature)
            remaining_votes = num_votes - np.sum(votes_per_feature)
            if remaining_votes > 0:
                votes_per_feature += (remaining_votes * (votes_per_feature / np.sum(votes_per_feature)))

        return grouped_features, votes_per_feature
    
    def dhondt_method(self, votes, num_mps):
        seats = np.zeros(len(votes), dtype=int)
        for _ in range(num_mps):
            quotients = votes / (seats + 1)
            max_index = np.argmax(quotients)
            seats[max_index] += 1
        return seats
    
    def plot_results(self, grouped_features, seats):
        plt.figure(figsize=(10, 6))
        plt.bar(grouped_features, seats)
        plt.xlabel("Feature/Alliance")
        plt.ylabel("Number of MPs (Importance)")
        plt.title("Feature Importance Using D'Hondt Method")
        plt.xticks(rotation=90)
        plt.show()
