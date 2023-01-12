import numpy as np
from matplotlib import pyplot as plt

# create noise data
def function(x, noise):
    y = np.sin(7*x+2) + noise
    return y

def function2(x, noise):
    y = np.sin(6*x+2) + noise
    return y

noise = np.random.uniform(low=-0.3, high=0.3, size=(100,))
x_line0 = np.linspace(1.95,2.85,100)
y_line0 = function(x_line0, noise)
x_line = np.linspace(0, 1.95, 100)
x_line2 = np.linspace(2.85, 3.95, 100)
x_pik = np.linspace(3.95, 5, 100)
y_pik = function2(x_pik, noise)
x_line3 = np.linspace(5, 6, 100)

# concatenate noise data
x = np.linspace(0, 6, 500)
y = np.concatenate((noise, y_line0, noise, y_pik, noise), axis=0)

# plot data
noise_band = 1.1
top_noise = y.mean()+noise_band*np.amax(noise)
bottom_noise = y.mean()-noise_band*np.amax(noise)
fig, ax = plt.subplots()
ax.axhline(y=y.mean(), color='red', linestyle='--')
ax.axhline(y=top_noise, linestyle='--', color='green')
ax.axhline(y=bottom_noise, linestyle='--', color='green')
ax.plot(x, y)

# split data into 2 signals
def split(arr, cond):
  return [arr[cond], arr[~cond]]

# find bottom noise data indexes
botom_data_indexes = np.argwhere(y < bottom_noise)
# split by visual x value
splitted_bottom_data = split(botom_data_indexes, botom_data_indexes < np.argmax(x > 3))

# find top noise data indexes
top_data_indexes = np.argwhere(y > top_noise)
# split by visual x value
splitted_top_data = split(top_data_indexes, top_data_indexes < np.argmax(x > 3))

# get first signal range
first_signal_start = np.amin(splitted_bottom_data[0])
first_signal_end = np.amax(splitted_top_data[0])

# get x index of first signal
x_first_signal = np.take(x, [first_signal_start, first_signal_end])
ax.axvline(x=x_first_signal[0], color='orange')
ax.axvline(x=x_first_signal[1], color='orange')

# get second signal range
second_signal_start = np.amin(splitted_top_data[1])
second_signal_end = np.amax(splitted_bottom_data[1])

# get x index of first signal
x_second_signal = np.take(x, [second_signal_start, second_signal_end])
ax.axvline(x=x_second_signal[0], color='orange')
ax.axvline(x=x_second_signal[1], color='orange')

plt.show()