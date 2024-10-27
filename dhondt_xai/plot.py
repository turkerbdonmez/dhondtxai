import matplotlib.pyplot as plt

def plot_dhondt_results(features, seats):
    plt.figure(figsize=(10, 6))
    plt.bar(features, seats)
    plt.xlabel("Feature/Alliance")
    plt.ylabel("Number of MPs (Importance)")
    plt.title("Feature Importance Using D'Hondt Method")
    plt.xticks(rotation=90)
    plt.show()
