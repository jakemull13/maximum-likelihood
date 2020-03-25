# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optim
import seaborn as sns

from itertools import product

# Always make it pretty.
plt.style.use('ggplot')

# %% [markdown]
# # Coinflips
# %% [markdown]
# 1

# %%
def flip_coin(n, p):
    """Flip a coin of fairness p, n times.
    
    Parameters
    ----------
    n: int
      The number of times to flip the coin.

    p: float, between zero and one.
      The probability the coin flips heads.

    Returns
    -------
    flips: np.array of ints
      The results of the coin flips, where 0 is a tail and 1 is a head.
    """
    return stats.binom(1,p).rvs(n)
    
flips = flip_coin(100, .6)

# %% [markdown]
# 2

# %%
def coin_log_likelihood(p, flips):
    """Return the log-likelihood of a parameter p given a sequence of coin flips.
    """
    heads=flips.sum()    
    l=heads*(np.log(p))+(len(flips)-heads)*(np.log(1-p))
    return l
coin_log_likelihood(.5,flips)

def coin_log_likelihood_2(p, flips):
    binomial = stats.binom(1, p)
    likelihoods = [binomial.pmf(datum) for datum in flips]
    return np.sum(np.log(likelihoods))
coin_log_likelihood_2(.5, flips)

# %% [markdown]
# 3

# %%
flip_data = [1, 0, 0, 0, 1, 1, 0, 0,0,0]


# %% [markdown]
# 4

# %%
log1 = coin_log_likelihood_2(.25, flip_data)
log2 = coin_log_likelihood_2(.50, flip_data)


# %% [markdown]
# 5

# %%
'''
labels = ['.25', '.5']
fig, ax = plt.subplots()
ax.bar([.25, .5],[log1, log2],)
'''

sns.barplot([.25, .5],[log1, log2])
# %% [markdown]
# 6

# %%
def plot_coin_likelihood(ax, ps, data):

    log1 = coin_log_likelihood_2(ps[0], data)
    log2 = coin_log_likelihood_2(ps[1], data)

    sns.barplot([ps[0], ps[1]], [log1, log2])

# %% [markdown]
# 7

# %%
fig, ax = plt.subplots()
for i in range(len(flip_data)):
    data = flip_data[0:i]
    plot_coin_likelihood(ax, [.25, .5], data)
# %% [markdown]
# 9

# %%
probabilities = list(np.linspace(0, 1, num=100))
ll = []
for p in probabilities:
    ll.append(coin_log_likelihood_2(p, flip_data))
ll
# %% 

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(probabilities, ll)
ax.plot([maximum_likelihood_estimate]*100,ll)





# %% 

maximum_likelihood_estimate = probabilities[np.argmax(ll)]
maximum_likelihood_estimate


# %%
