import numpy as np

def process_exclusions(X, exclude_features):
    return X.drop(columns=[col for col in exclude_features if col in X.columns])

def process_alliances(features, importances, alliances):
    alliance_importances = {}
    grouped_importances = []
    grouped_features = []
    used_features = set()
    
    for alliance in alliances:
        total_importance = sum(importances[features.index(var.strip())] for var in alliance if var.strip() in features)
        if total_importance > 0:
            alliance_name = ' + '.join([var.strip() for var in alliance])
            alliance_importances[alliance_name] = total_importance
            used_features.update([var.strip() for var in alliance])
    
    for alliance_name, importance in alliance_importances.items():
        grouped_importances.append(importance)
        grouped_features.append(alliance_name)
    
    for feature, importance in zip(features, importances):
        if feature not in used_features:
            grouped_importances.append(importance)
            grouped_features.append(feature)
    
    return np.array(grouped_importances), grouped_features

def d_hondt_method(votes, total_deputies):
    seats = np.zeros(len(votes), dtype=int)
    for _ in range(total_deputies):
        quotients = votes / (seats + 1)
        max_index = np.argmax(quotients)
        seats[max_index] += 1
    return seats
