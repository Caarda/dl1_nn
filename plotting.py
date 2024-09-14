import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, ax1):
    ax1.clear()
    ax1.set_title("Scores over Time")
    ax1.set_xlabel("n")
    ax1.set_ylabel("dist")
    ax1.plot(scores)
    ax1.plot(mean_scores)
    ax1.set_ylim(ymin=0)
    ax1.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    ax1.legend()

def plot2(flight, best_flight, ax2):
    ax2.clear()
    ax2.set_title("Flight Path")
    ax2.set_xlabel("dist")
    ax2.set_ylabel("y")
    f_dist, f_y = zip(*flight)
    b_dist, b_y = zip(*best_flight)
    ax2.plot(f_dist, f_y, label="Current Flight")
    ax2.plot(b_dist, b_y, label="Best Flight")
    ax2.set_ylim(ymin=0)
    ax2.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

def updateplots(scores, mean_scores, flight, best_flight):
    plot(scores, mean_scores, ax1)
    plot2(flight, best_flight, ax2)
    plt.show(block=False)
    plt.pause(.1)