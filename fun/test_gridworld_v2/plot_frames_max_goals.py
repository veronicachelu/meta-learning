import matplotlib.pyplot as plt

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

y = [0 for _ in range(10)]
y[0] = 1
y[4] = 2
y[7] = 3
y[8] = 1
y[9] = 3
N = len(y)
x = range(N)
width = 1/1.5
plt.bar(x, y, width, color="blue")

plt.savefig('nonmatching_max_goals.png')
# fig = plt.gcf()
# plot_url = py.plot_mpl(fig, filename='nonmatching_max_goals')