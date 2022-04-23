import numpy as np
from math import e, log, sqrt, pi
from scipy.stats import norm
import matplotlib.pyplot as plt
import itertools

def buildTree(S, vol, T, N):
    dt = T / N

    matrix = np.zeros((N + 1, N + 1))

    u = e ** (vol * np.sqrt(dt))
    d = e ** (-vol * np.sqrt(dt))

    # Iterate over the lower triangle
    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            # Hint: express each cell as a combination of up and down moves
            ##matrix[i,j] = 0 ## TODO
            matrix[i, j] = S * (u ** j) * (d ** (i - j))

    return matrix


def valueOptionMatrix(tree, T, N, r, K, vol, opttype, region):
    # Parameters
    dt = T / N
    u = e ** (vol * np.sqrt(dt))
    d = e ** (-vol * np.sqrt(dt))
    p = ((e ** (r * dt)) - d) / (u - d)
    # Tree
    option_tree = np.zeros(tree.shape)
    columns = tree.shape[1]
    rows = tree.shape[0]

    # Walk backward, to start at the last row of the matrix

    # Add pay-off function in the last row
    for c in np.arange(columns):
        S = tree[rows - 1, c]  # value in the matrix

        if opttype == 'call':
            option_tree[rows - 1, c] = np.max([S - K, 0])
        elif opttype == "put":
            option_tree[rows - 1, c] = np.max([K - S, 0])

    # For other rows, combine with previous rows. Walk backwards, from last row to first row
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = option_tree[i + 1, j]
            up = option_tree[i + 1, j + 1]
            if region == "europe":
                option_tree[i, j] = (e ** (-r * dt)) * ((p * up) + ((1 - p) * down))
            elif region == "america" and opttype == "call":
                option_tree[i, j] = max((e ** (-r * dt)) * ((p * up) + ((1 - p) * down)), tree[i, j] - K)
            elif region == "america" and opttype == "put":
                option_tree[i, j] = max((e ** (-r * dt)) * ((p * up) + ((1 - p) * down)), K - tree[i, j])
            else:
                print("Wrong parameters specified.")

    return option_tree


def d1(S, K, T, r, sigma):
    return ((log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * sqrt(T)))


def d2(S, K, T, r, sigma):
    return (d1(S, K, T, r, sigma) - sigma * sqrt(T))


def bs_call(S, K, T, r, sigma):
    return (S * norm.cdf(d1(S, K, T, r, sigma)) - (K * e ** (-r * T) * norm.cdf(d2(S, K, T, r, sigma))))


def bs_put(S, K, T, r, sigma):
    return (K * e ** (-r * T) - S + bs_call(S, K, T, r, sigma))


def bs_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


# Parameters
sigma = 0.2
S = 100
T = 1.0
N = 50
K = 99
r = 0.06
opttype = "call"

# Calculate Stock Prices with Binomial Model
stocktree = buildTree(S, sigma, T, N)
print("----Binomial Stock Prices----")
print(stocktree)
# Calculate Eurpean Option Prices with Binomial Model
eu_option_tree = valueOptionMatrix(stocktree, T, N, r, K, sigma, opttype, region="europe")
print("----Binomial Option Prices----")
print(eu_option_tree)
# Calculate American Option Prices with Binomial Model
us_option_tree = valueOptionMatrix(stocktree, T, N, r, K, sigma, opttype, region="america")
print("----Binomial American Option Prices----")
print(us_option_tree)
# Calculate European Option Prices with Black-Scholes Model
optionPriceAnalytical = bs_call(S, K, T, r, sigma)
print("Black Scholes Option Price: " + str(optionPriceAnalytical))
# Calculate European Delta with Binomial Model
print("Binomial Tree Model Delta: ",
      (eu_option_tree[1, 1] - eu_option_tree[1, 0]) / (stocktree[1, 1] - stocktree[1, 0]))
# Calculate European Delta with Black-Scholes Model
print("Black Scholes Delta: " + str(bs_delta(S, K, T, r, sigma)))

# Plotting
nsteps = np.arange(2, 200, 5)
black_scholes = np.zeros(nsteps.shape)
black_scholes += bs_call(S, K, T, r, sigma)
binomial_convergence = [valueOptionMatrix(buildTree(S, sigma, T, steps),
                                          T, steps, r, K, sigma, opttype, region="europe")[0, 0] for steps in nsteps]
plt.plot(nsteps, black_scholes, label="Black-Scholes")
plt.plot(nsteps, binomial_convergence, '--r', label="Binomial")
plt.legend(loc="upper center")
plt.title("Binomial Option Price with Number of Steps.")
plt.xlabel("# Steps")
plt.ylabel("Option Price")
plt.show()

# Volatility
x = np.linspace(0.01, 1, 50)  # Volatility Space
binom_vals = []
for i in x:
    binom_vals.append(valueOptionMatrix(buildTree(S, i, T, N), T, N, r, K, i, opttype, region="europe")[0, 0])
plt.plot(x, bs_call(S, K, T, r, x), label="Black-Scholes")
plt.plot(x, binom_vals, '--r', label="Binomial")
plt.legend(loc="upper center")
plt.title("Volatility Impact on Binomial and Black-Scholes Options Price")
plt.xlabel("Volatility")
plt.ylabel("Options Value")
plt.show()

# Hedge parameter (Delta) vs Volatility
x = np.linspace(0.01, 1, 50)  # Volatility Space
binom_delta = []
black_scholes_delta = []
for i in x:
    binomstock = buildTree(S, i, T, N)
    binomoption = valueOptionMatrix(binomstock, T, N, r, K, i, opttype, region="europe")
    delta = (binomoption[1, 1] - binomoption[1, 0]) / (binomstock[1, 1] - binomstock[1, 0])
    binom_delta.append(delta)
# Plot
plt.plot(x, bs_delta(S, K, T, r, x), label="Black-Scholes")
plt.plot(x, binom_delta, '--r', label="Binomial")
plt.legend(loc="upper center")
plt.title("Binomial vs. Black Scholes Delta Value subject to increasing Volatility")
plt.xlabel("Volatility")
plt.ylabel("Delta Value")
plt.show()

# Hedging simulation
print("-----Euler Simulation-----")
stock_sigma = 0.4
bs_sigma = 0.2
M = np.arange(1, 365)  # Daily Steps
dt = T / len(M)
runs = np.arange(0, 20, 1)
stock_runs = []
for run in runs:
    stock_vals = [S]
    for m in M:
        zm = np.random.normal()
        Sm = stock_vals[m - 1] + (r * stock_vals[m - 1] * dt) + (stock_sigma * stock_vals[m - 1] * zm * sqrt(dt))
        stock_vals.append(Sm)
    stock_runs.append(stock_vals)

# Plot Stock Simulation
plt.title(" Daily Stock Simulation with 10% Volatility")
plt.xlabel("Time Steps")
plt.ylabel("Price of Stock in EUROS")
for i in range(len(stock_runs)):
    plt.plot(stock_runs[i])
plt.show()

# Calculate Hedge Parameter
delta_runs = []
for stock_run in stock_runs:
    deltam = []
    for sm in stock_run:
        deltam.append(bs_delta(sm, K, T, r, bs_sigma))
    delta_runs.append(deltam)

print("---------------")

# Hedging Parameter value at each timestep
for delta_run in delta_runs:
    #deltadiff = np.diff(deltam)
    plt.plot(delta_run)
plt.title("Hedging Parameter")
plt.show()
# Hedging Parameter difference with the previous -> cover exposure
delta_diffs = []
for delta_run in delta_runs:
    deltadiff = np.diff(delta_run)
    delta_diffs.append(deltadiff)
    plt.plot(deltadiff)
plt.title("Hedging Parameter Difference")
plt.show()

# Hedging Difference Distribution
all_differences = list(itertools.chain.from_iterable(delta_diffs))
plt.hist(all_differences, bins=200, density=False)
plt.title("Delta Hedging differences Distribution")
plt.show()
