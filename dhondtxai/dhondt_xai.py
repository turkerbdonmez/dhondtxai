
import numpy as np
import matplotlib.pyplot as plt

class DhondtXAI:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.feature_importances = None
        self.correlation_info = None
    
    def fit(self, X_train, y_train):  # Include y_train to fit the model
        # Train the provided model and extract feature importances
        self.model.fit(X_train, y_train)
        self.features = list(X_train.columns)  # Convert to a list to avoid pandas Index issues
        self.feature_importances = self.model.feature_importances_
        
        # Calculate correlation between features and the main target variable
        correlations = X_train.corrwith(y_train).to_dict()
        self.correlation_info = {feature: correlations.get(feature, 0) for feature in self.features}
    
    def select_features(self, feature_names):
        print("Available features:")
        for idx, feature in enumerate(feature_names):
            print(f"{idx + 1}. {feature}")

        # Allow user to specify features to exclude from the evaluation
        exclude_input = input("Enter the variables you want to exclude from the evaluation (e.g., '2, 4' or 'var2, var4') or 'none': ")
        exclude_features = []
        if exclude_input.lower() != 'none':
            exclude_parts = exclude_input.split(',')
            for part in exclude_parts:
                var = part.strip()
                if var.isdigit():
                    feature_index = int(var) - 1
                    if 0 <= feature_index < len(feature_names):
                        exclude_features.append(feature_names[feature_index])
                else:
                    exclude_features.append(var)

        # Allow user to enter alliances using feature names or index numbers
        alliances_input = input("Enter any alliances between the variables (e.g., '2,3 and 4' or 'var2, var3 and var4') or 'none': ")
        alliances = []
        if alliances_input.lower() != 'none':
            alliance_parts = alliances_input.split(', ')
            for part in alliance_parts:
                variables = [var.strip() for var in part.split(' and ')]
                processed_vars = []
                for var in variables:
                    if var.isdigit():
                        feature_index = int(var) - 1
                        if 0 <= feature_index < len(feature_names):
                            processed_vars.append(feature_names[feature_index])
                    else:
                        if var in feature_names:
                            processed_vars.append(var)
                if processed_vars:
                    alliances.append(processed_vars)
        
        return alliances, exclude_features

    
    def apply_dhondt(self, num_votes, num_mps, threshold=None, alliances=None, exclude_features=None):
        # Exclude specified features if provided
        features = [f for f in self.features if f not in exclude_features] if exclude_features else self.features
        
        # Allow dynamic alliance creation from feature names
        feature_importances = self.feature_importances[:len(features)]
        grouped_importances = []
        grouped_features = []
        used_features = set()

        if alliances:
            for alliance in alliances:
                # Convert alliance names to feature names if provided as list of lists
                alliance_features = [var.strip() for var in alliance if var.strip() in features]
                if alliance_features:
                    total_importance = sum(self.feature_importances[features.index(var)] for var in alliance_features)
                    if total_importance > 0:
                        alliance_name = ' + '.join(alliance_features)
                        grouped_features.append(alliance_name)
                        grouped_importances.append(total_importance)
                        used_features.update(alliance_features)

        # Add remaining individual features not in any alliance
        for feature, importance in zip(features, feature_importances):
            if feature not in used_features:
                grouped_features.append(feature)
                grouped_importances.append(importance)

        grouped_importances = np.array(grouped_importances)
        
        # Calculate votes per feature/alliance
        votes_per_feature = (grouped_importances / grouped_importances.sum()) * num_votes

        # Apply threshold if specified, but do not zero out votes; instead, track excluded features
        if threshold is not None:
            threshold_votes = (threshold / 100) * num_votes
            excluded_features = np.where(votes_per_feature < threshold_votes, True, False)
        else:
            excluded_features = np.full_like(votes_per_feature, False, dtype=bool)

        return grouped_features, votes_per_feature, excluded_features

        # Exclude specified features if provided
        features = [f for f in self.features if f not in exclude_features] if exclude_features else self.features
        
        # Allow dynamic alliance creation from feature names
        feature_importances = self.feature_importances[:len(features)]
        grouped_importances = []
        grouped_features = []

        used_features = set()

        if alliances:
            for alliance in alliances:
                # Convert alliance names to feature names if provided as list of lists
                alliance_features = [var.strip() for var in alliance if var.strip() in features]
                if alliance_features:
                    # Convert features to a list and use index lookup safely
                    total_importance = sum(self.feature_importances[features.index(var)] for var in alliance_features)
                    if total_importance > 0:
                        alliance_name = ' + '.join(alliance_features)
                        grouped_features.append(alliance_name)
                        grouped_importances.append(total_importance)
                        used_features.update(alliance_features)
        
        # Add remaining individual features not in any alliance
        for feature, importance in zip(features, feature_importances):
            if feature not in used_features:
                grouped_features.append(feature)
                grouped_importances.append(importance)
        
        grouped_importances = np.array(grouped_importances)
        
        # Calculate votes per feature/alliance
        votes_per_feature = (grouped_importances / grouped_importances.sum()) * num_votes

        # Apply threshold if specified, otherwise handle as None
        if threshold is not None:
            threshold_votes = (threshold / 100) * num_votes
            votes_per_feature = np.where(votes_per_feature < threshold_votes, 0, votes_per_feature)
            remaining_votes = num_votes - np.sum(votes_per_feature)
            if remaining_votes > 0:
                votes_per_feature += (remaining_votes * (votes_per_feature / np.sum(votes_per_feature)))

        return grouped_features, votes_per_feature
    
    
    def dhondt_method(self, votes, num_mps, excluded_features):
        seats = np.zeros(len(votes), dtype=int)
        for _ in range(num_mps):
            quotients = np.where(excluded_features, -1, votes / (seats + 1))  # Set excluded feature quotients to -1 to ignore them
            max_index = np.argmax(quotients)
            seats[max_index] += 1
        return seats

        seats = np.zeros(len(votes), dtype=int)
        for _ in range(num_mps):
            quotients = votes / (seats + 1)
            max_index = np.argmax(quotients)
            seats[max_index] += 1
        return seats
    
    def plot_results(self, grouped_features, seats):
        # Use correlation info to determine color
        colors = []
        for feature in grouped_features:
            if ' + ' in feature:  # For alliances, use the first feature's correlation
                main_feature = feature.split(' + ')[0].strip()
            else:
                main_feature = feature
            
            correlation = self.correlation_info.get(main_feature, 0)
            color = 'blue' if correlation >= 0 else 'red'
            colors.append(color)
        
        # Plot results with color-coded bars
        plt.figure(figsize=(10, 6))
        plt.bar(grouped_features, seats, color=colors)
        plt.xlabel("Feature/Alliance")
        plt.ylabel("Number of MPs (Importance)")
        plt.title("Feature Importance Using D'Hondt Method (Color-Coded by Correlation)")
        plt.xticks(rotation=90)
        plt.show()
