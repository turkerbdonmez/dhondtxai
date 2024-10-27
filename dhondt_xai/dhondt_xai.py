
import numpy as np

class DHondtXAI:
    def __init__(self, model, X_train, num_votes, num_mps, threshold_percentage=None, exclude_features=None, alliances=None):
        self.model = model
        self.X_train = X_train
        self.num_votes = num_votes
        self.num_mps = num_mps
        self.threshold_percentage = threshold_percentage
        self.exclude_features = exclude_features or []
        self.alliances = alliances or []
        self.feature_importances = None
        self.grouped_features = []
        self.votes_per_feature = None

    def train_model(self):
        # Remove excluded features
        X_train = self.X_train.drop(columns=[col for col in self.exclude_features if col in self.X_train.columns])
        self.model.fit(X_train)
        self.feature_importances = self.model.feature_importances_
        self.features = X_train.columns.tolist()

    def process_alliances(self):
        alliance_importances = {}
        alliance_names = []
        used_features = set()

        for alliance in self.alliances:
            total_importance = sum(self.feature_importances[self.features.index(var.strip())] 
                                   for var in alliance if var.strip() in self.features)
            if total_importance > 0:
                alliance_name = ' + '.join([var.strip() for var in alliance])
                alliance_names.append(alliance_name)
                alliance_importances[alliance_name] = total_importance
                used_features.update([var.strip() for var in alliance])

        grouped_importances = []
        grouped_features = []

        for alliance_name in alliance_names:
            grouped_importances.append(alliance_importances[alliance_name])
            grouped_features.append(alliance_name)

        for feature, importance in zip(self.features, self.feature_importances):
            if feature not in used_features:
                grouped_importances.append(importance)
                grouped_features.append(feature)

        self.grouped_features = grouped_features
        self.votes_per_feature = (np.array(grouped_importances) / sum(grouped_importances)) * self.num_votes

    def apply_threshold(self):
        if self.threshold_percentage is not None:
            threshold_votes = (self.threshold_percentage / 100) * self.num_votes
            self.votes_per_feature = np.where(self.votes_per_feature < threshold_votes, 0, self.votes_per_feature)
            remaining_votes = self.num_votes - np.sum(self.votes_per_feature)
            if remaining_votes > 0:
                self.votes_per_feature += (remaining_votes * (self.votes_per_feature / np.sum(self.votes_per_feature)))

    def d_hondt_method(self):
        seats = np.zeros(len(self.votes_per_feature), dtype=int)
        for _ in range(self.num_mps):
            quotients = self.votes_per_feature / (seats + 1)
            max_index = np.argmax(quotients)
            seats[max_index] += 1
        return seats

    def calculate_seats(self):
        self.train_model()
        self.process_alliances()
        self.apply_threshold()
        seats = self.d_hondt_method()
        return dict(zip(self.grouped_features, seats))
