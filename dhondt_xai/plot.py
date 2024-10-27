
import matplotlib.pyplot as plt

def plot_dhondt_results(seats, title="Feature Importance Using D'Hondt Method"):
    features, seat_counts = zip(*seats.items())
    plt.figure(figsize=(10, 6))
    plt.bar(features, seat_counts)
    plt.xlabel("Feature/Alliance")
    plt.ylabel("Number of MPs (Importance)")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()
