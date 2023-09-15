import os
import matplotlib.pyplot as plt
from IPython import display

"""This code defines a plot function that is used to dynamically generate
 and display a score graph while training an agent"""

# Initialize matplotlib interactive mode
# plt.ion()

def plot(scores, mean_scores):
    # Clear the previous output and display the new graph in interactive mode
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Set axis title and labels
    plt.title("Training..")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot scores and average scores
    plt.plot(scores)
    plt.plot(mean_scores)

    # Imposta il limite inferiore dell'asse y a 0
    plt.ylim(ymin=0)

    # Add texts with the latest scores to the end of the graphs
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # Save the graph img
    if os.path.exists("model/score_plot.png"):
        plt.savefig("model/score_plot.png")
    else:
        plt.savefig("model/score_plot.png")