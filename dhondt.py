# dhondt.py
import numpy as np

class DhondtXAI:
    def __init__(self, feature_importances, features):
        self.feature_importances = feature_importances
        self.features = features

    def get_votes(self, num_votes, alliances=None, exclude_features=None, threshold=None):
        if exclude_features:
            self.features = [f for f in self.features if f not in exclude_features]
            self.feature_importances = [
                imp for imp, f in zip(self.feature_importances, self.features) if f not in exclude_features
            ]
        
        alliance_importances = {}
        used_features = set()
        grouped_features = []
        grouped_importances = []

        if alliances:
            for alliance in alliances:
                total_importance = sum(
                    self.feature_importances[self.features.index(var.strip())] for var in alliance if var.strip() in self.features
                )
                if total_importance > 0:
                    alliance_name = ' + '.join([var.strip() for var in alliance])
                    alliance_importances[alliance_name] = total_importance
                    used_features.update([var.strip() for var in alliance])

        for name, importance in alliance_importances.items():
            grouped_features.append(name)
            grouped_importances.append(importance)

        for feature, importance in zip(self.features, self.feature_importances):
            if feature not in used_features:
                grouped_features.append(feature)
                grouped_importances.append(importance)

        grouped_importances = np.array(grouped_importances)
        votes_per_feature = (grouped_importances / grouped_importances.sum()) * num_votes

        if threshold:
            threshold_votes = (threshold / 100) * num_votes
            votes_per_feature = np.where(votes_per_feature < threshold_votes, 0, votes_per_feature)

            remaining_votes = num_votes - np.sum(votes_per_feature)
            if remaining_votes > 0:
                votes_per_feature = votes_per_feature + (remaining_votes * (votes_per_feature / np.sum(votes_per_feature)))

        return dict(zip(grouped_features, votes_per_feature))

    def dhondt_allocation(self, votes, total_deputies):
        seats = np.zeros(len(votes), dtype=int)
        for _ in range(total_deputies):
            quotients = votes / (seats + 1)
            max_index = np.argmax(quotients)
            seats[max_index] += 1
        return seats
