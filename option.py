import random
import numpy as np
from math import e, log, sqrt, pi
from scipy.stats import norm
from abc import ABCMeta, abstractmethod, ABC


class Option(metaclass=ABCMeta):
    """
    Class to represent an option.
    ...
    Attributes
    ----------
    s: float
        current stock price of the product
    k: float
        strike price
    t: int
        maturity of the option or delivery date
    r: float
        interest rate
    sigma: float
        volatility
    opt_type: string
        either 'call' or 'put' indicating the option type.

    Methods:
    -------
    binomial_model_overview(n_steps):
        - calculate and print option price using binomial tree.
        - calculate and print hedge positions for the binomial tree option prices.
    """

    def __init__(self, s, k, t, r, sigma, opt_type):
        """
        Constructor for Option class.
        :param s: float
            stock price
        :param k: float
            strike
        :param t: int
            maturity
        :param r: float
            interest rate
        :param sigma: float
            volatility
        :param opt_type: str
            option type; can be either 'call' or 'put'.
        """
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.opt_type = opt_type

    def calculate_payoff(self, s_t):
        """
        Calculates payoff based on option type.
        :param s_t: stock price values at maturity t.
        :return: intrinsic payoff value.
        """
        if self.opt_type == 'call':
            payoff = np.clip(s_t - self.k, 0, None)
        elif self.opt_type == "put":
            payoff = np.clip(self.k - s_t, 0, None)
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            payoff = None
        return payoff

    def binomial_model_overview(self, n_steps):
        """
        Calculate and print option price using binomial tree.
        :param n_steps: int
            number of steps to consider in the binomial model.
        :return: option price (f) and hedging position (delta)
        """
        # simulate asset prices
        asset_prices = self.binomial_asset_prices(n_steps)
        # calculate option prices
        f, option_prices_matrix = self.binomial_option_prices(asset_prices)
        delta, hedged_positions_matrix = self.binomial_hedged_positions(asset_prices, option_prices_matrix)
        print_model = "| Binomial Model ".ljust(40, '-')
        print_price = f' {f:.4f} EUR'.rjust(10, '-')
        print_hedge = f' | ∆: {delta:.4f} shares'.rjust(10, '-')
        print(print_model, print_price, print_hedge)
        return f, delta

    def binomial_asset_prices(self, n_steps):
        """
        Simulate asset price in a two-state economy (binomial model) for 'n' time steps.
        :param n_steps: int
            number of steps to consider in the binomial model.
        :return: np.array
            (n_steps x n_steps) matrix with binomial asset prices
        """
        dt = self.t / n_steps  # time interval
        matrix = np.zeros((n_steps + 1, n_steps + 1))  # initialize matrix
        u = e ** (self.sigma * np.sqrt(dt))  # up factor
        d = e ** (-self.sigma * np.sqrt(dt))  # down factor
        # Generate asset prices with binomial model
        for i in np.arange(n_steps + 1):
            for j in np.arange(i + 1):
                # Express each cell as a combination of up and down moves
                matrix[j, i] = self.s * (u ** (i - j)) * (d ** (j))
        return matrix

    @abstractmethod
    def binomial_option_prices(self, asset_prices):
        pass

    def binomial_hedged_positions(self, asset_prices, option_prices):
        """
        Calculate hedging levels for each time element of the binomial tree.
        :param asset_prices: np.array
            Price evolution of the underlying asset.
        :param option_prices: np.array
            Price evolution the the option price.
        :return: float, np.array
            Hedge value for t=0 (hedge) and matrix with hedging positions (hedged_positions)
        """
        hedged_positions = np.zeros(asset_prices.shape)
        n_steps = len(asset_prices) - 1
        for i in range(n_steps, 0, -1):
            f_u, f_d = option_prices[:i, i], option_prices[1:i + 1, i]
            s_u, s_d = asset_prices[:i, i], asset_prices[1:i + 1, i]
            hedge = (f_u - f_d) / (s_u - s_d)
            hedged_positions[:i, i - 1] = hedge
        # retrieve hedged position today
        hedged_positions[0, 0] = (option_prices[0, 1] - option_prices[1, 1])/(asset_prices[0, 1] - asset_prices[1, 1])
        return hedged_positions[0, 0], hedged_positions


class EuropeanOption(Option, ABC):

    def __init__(self, s, k, t, r, sigma, opt_type):
        super().__init__(s, k, t, r, sigma, opt_type)
        self.opt_style = 'european'

    def binomial_option_prices(self, asset_prices):
        """
        Calculates option prices for the different elements of the binomial model.
        :param asset_prices: np.array
        :return: float, np.array
        """
        # calculate payoffs at time t
        payoffs_t = self.calculate_payoff(asset_prices[:, -1])
        # initialise option price matrix with payoffs at time t
        option_prices = np.zeros(asset_prices.shape)
        option_prices[:, -1] = payoffs_t
        # binomial model parameters
        n_steps = len(asset_prices) - 1
        dt = self.t / n_steps
        u = e ** (self.sigma * np.sqrt(dt))
        d = e ** (-self.sigma * np.sqrt(dt))
        p = ((e ** (self.r * dt)) - d) / (u - d)
        # bring back payoffs to time 0
        for i in range(n_steps, 0, -1):
            # select upper and lower option prices of the matrix at time t
            f_u = option_prices[:i, i]
            f_d = option_prices[1:i + 1, i]
            # calculate option prices at time t-1
            f = (e ** (-self.r * dt)) * ((p * f_u) + ((1 - p) * f_d))
            # replace in option price matrix
            option_prices[:i, i - 1] = f
        # retrieve option fair price t=0
        option_prices[0, 0] = (e ** (-self.r * dt)) * ((p * option_prices[0, 1]) + ((1 - p) * option_prices[1, 1]))
        return option_prices[0, 0], option_prices


    def blackscholes_price(self):
        """
        Black-Scholes calculation of the option price and the hedged position.
        :return: float, float
            option price (f) and hedge position in amount of shares (delta)
        """
        f = -1
        if self.opt_type == 'call':
            f = self.__bs_call()
        elif self.opt_type == "put":
            f = self.__bs_put()
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            option_value = None
        delta = self.__bs_delta()
        print_model = "| Black-Scholes Model ".ljust(40, '-')
        print_price = f' {f:.4f} EUR'.rjust(10, '-')
        print_hedge = f' | ∆: {delta:.4f} shares'.rjust(10, '-')
        print(print_model, print_price, print_hedge)
        return f, delta

    def __d1(self):
        return (log(self.s / self.k) + (self.r + self.sigma ** 2 / 2.) * self.t) / (self.sigma * sqrt(self.t))

    def __d2(self):
        return self.__d1() - self.sigma * sqrt(self.t)

    def __bs_call(self):
        return self.s * norm.cdf(self.__d1()) - (self.k * e ** (-self.r * self.t) * norm.cdf(self.__d2()))

    def __bs_put(self):
        return self.k * e ** (-self.r * self.t) - self.s + self.__bs_call()

    def __bs_delta(self):
        if self.opt_type == "call":
            delta = norm.cdf(self.__d1())
        elif self.opt_type == "put":
            delta = norm.cdf(self.__d1()) - 1
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            delta = None
        return delta


class AmericanOption(Option, ABC):

    def __init__(self, s, k, t, r, sigma, opt_type):
        super().__init__(s, k, t, r, sigma, opt_type)
        self.opt_style = 'american'

    def binomial_option_prices(self, asset_prices):
        """
        Calculates option prices for the different elements of the binomial model.
        :param asset_prices: np.array
        :return: float, np.array
        """
        # calculate payoffs at time t
        payoffs_t = self.calculate_payoff(asset_prices[:, -1])
        # initialise option price matrix with payoffs at time t
        option_values = np.zeros(asset_prices.shape)
        option_values[:, -1] = payoffs_t
        # binomial model parameters
        n_steps = len(asset_prices) - 1
        dt = self.t / n_steps
        u = e ** (self.sigma * np.sqrt(dt))
        d = e ** (-self.sigma * np.sqrt(dt))
        p = ((e ** (self.r * dt)) - d) / (u - d)
        # bring back payoffs to time 0
        for i in range(n_steps, 0, -1):
            # select upper and lower option prices of the matrix at time t
            f_u = option_values[:i, i]
            f_d = option_values[1:i + 1, i]
            # calculate option prices at time t-1
            f = (e ** (-self.r * dt)) * ((p * f_u) + ((1 - p) * f_d))
            # compare with intrinsic value and choose maximum (account for early exercise opportunity)
            f = np.maximum(f, self.calculate_payoff(asset_prices[:i, i - 1]))
            # replace in option price matrix
            option_values[:i, i - 1] = f
        # retrieve option fair price t=0
        option_values[0, 0] = (e ** (-self.r * dt)) * ((p * option_values[0, 1]) + ((1 - p) * option_values[1, 1]))
        option_values[0, 0] = np.maximum(option_values[0, 0], self.calculate_payoff(asset_prices[0, 0]))
        return option_values[0, 0], option_values


if __name__ == "__main__":
    # Initialize option object
    print('American Option:')
    american_opt = AmericanOption(s=100, k=120, t=20.0, r=0.00, sigma=0.01, opt_type='put')
    american_opt.binomial_model_overview(n_steps=3)
    print('European Option:')
    european_opt = EuropeanOption(s=100, k=120, t=20.0, r=0.0, sigma=0.01, opt_type='put')
    european_opt.binomial_model_overview(n_steps=20)
    european_opt.blackscholes_price()
    print('European Option:')
    european_opt = EuropeanOption(s=100, k=80, t=20.0, r=0.0, sigma=0.01, opt_type='call')
    european_opt.binomial_model_overview(n_steps=21)
    european_opt.blackscholes_price()