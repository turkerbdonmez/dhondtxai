import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from dhondt_xai.utils import process_exclusions, process_alliances, d_hondt_method

class DhondtXAI:
    def __init__(self, model_type='xgboost', num_votes=100, num_mps=10, threshold=None):
        self.model_type = model_type
        self.model = None
        self.num_votes = num_votes
        self.num_mps = num_mps
        self.threshold = threshold
    
    def train_model(self, X_train, y_train):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier()
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier()
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(eval_metric='logloss')
        elif self.model_type == 'catboost':
            self.model = CatBoostClassifier(verbose=0)
        else:
            raise ValueError("Unsupported model type.")
        
        self.model.fit(X_train, y_train)

    def calculate_importances(self, X_train, exclude_features, alliances):
        features = X_train.columns.tolist()
        X_train = process_exclusions(X_train, exclude_features)
        
        importances = self.model.feature_importances_
        grouped_importances, grouped_features = process_alliances(features, importances, alliances)
        
        return self.apply_dhondt(grouped_importances, grouped_features)

    def apply_dhondt(self, importances, features):
        votes_per_feature = (importances / importances.sum()) * self.num_votes
        
        if self.threshold is not None:
            threshold_votes = (self.threshold / 100) * self.num_votes
            votes_per_feature = np.where(votes_per_feature < threshold_votes, 0, votes_per_feature)
            remaining_votes = self.num_votes - np.sum(votes_per_feature)
            if remaining_votes > 0:
                votes_per_feature += (remaining_votes * (votes_per_feature / np.sum(votes_per_feature)))
        
        seats = d_hondt_method(votes_per_feature, self.num_mps)
        
        return dict(zip(features, seats))
