# For Graph Plotting
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
start = time.time()

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    # Add x and y to lists
    # xs.append(time.ctime())
    xs.append(time.ctime())
    with open("distance.txt", "r") as f:
        ys.append(float(f.read()))

    # Limit x and y lists to 20 items
    # xs = xs[-20:]
    # ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title("Distance Over Time")
    plt.ylabel('Distance')
    plt.xlabel('Time')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
plt.show()
