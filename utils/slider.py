import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create a figure and an axis to plot the data on
fig, ax = plt.subplots()

# Create the data that you want to plot
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Plot the data on the axis
ax.plot(data)

# Add the slider widget to the figure
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(slider_ax, 'Date', 0, len(data), valinit=0, valstep=1)

# Define a function to update the plot when the slider is moved
def update(val):
    window_start = int(slider.val)
    window_end = window_start + 1
    ax.set_xlim(window_start, window_end)
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Display the plot
plt.show()
