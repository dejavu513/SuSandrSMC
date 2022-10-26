import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

###initialization
N = 300  ###number of samples
T = 30  ###number of time steps

####Hidden Markov model(stochastic volatility model)
rho: float = 0.95###
sigma = 1 ##
beta: float = 0.5###
initiVar = (sigma ** 2) / (1 - rho ** 2)  ##initial density

x0 = np.random.normal(0, math.sqrt(initiVar), 1)
true_x = np.zeros([T, 1])  ###hidden states
y = np.zeros([T, 1]) ###observations
true_x[0] = x0
y[0] = beta * math.exp(true_x[0] / 2) * np.random.normal(0, 1, 1)

###true tracjatory from
for t in range(1, T):
    true_x[t] = rho * true_x[t - 1] + sigma * np.random.normal(0, 1, 1)
    y[t] = beta * math.exp(true_x[t - 1] / 2) * np.random.normal(0, 1, 1)

####rejection SMC
x = np.random.rand(N, T)  ###sampling particles
xu = np.random.rand(N, T)  ####particles after resampling
q = np.random.rand(N, T)  ###normalised weights
qq = np.random.rand(N, T)  ###unnormalised weights
I = np.random.rand(N, T)  ###offsprings
R = np.random.rand(N, 1)  ###variance
#result = np.random.rand(T, 1)
x[:, 0] = np.random.normal(0, math.sqrt(initiVar), N)
for j in range(1, N + 1):
    R[j - 1] = (beta ** 2) * math.exp(x[j - 1, 0])
for j in range(1, N + 1):
    qq[j - 1, 0] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[0] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
q[:, 0] = qq[:, 0] / math.fsum(qq[:, 0])
probs = q[:, 0].tolist()
max_weights = max(q[:, 0])
for j in range(1, N + 1):
    thre = np.random.uniform(0, 1)
    b = q[j - 1, 0] / max_weights
    if thre >= b:
        A = np.random.multinomial(1, probs)
        A = A.tolist()
        A = A.index(1)
        xu[j - 1, 0] = x[A, 0]
###update and prediction stages
for t in range(1, T):
    x[:, t] = rho * xu[:, t - 1] + sigma * np.random.normal(0, 1, N)
    for j in range(1, N + 1):
        R[j - 1] = (beta ** 2) * math.exp(x[j - 1, t])
    for j in range(1, N + 1):
        qq[j - 1, t] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[t] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
    q[:, t] = qq[:, t] / sum(qq[:, t])
    ###resampling step
    probs = q[:, t].tolist()
    max_weights = max(q[:, t])
    for j in range(1, N + 1):
        thre = np.random.uniform(0, 1)
        b = q[j - 1, t] / max_weights
        if thre >= b:
            A = np.random.multinomial(1, probs)
            A = A.tolist()
            A = A.index(1)
            xu[j - 1, t] = x[A, t]
            xu[j - 1, 0:t] = xu[A, 0:t]
        else:
            xu[j - 1, t] = x[j - 1, t]

xaxs = range(1, T + 1)
for i in range(1, N+1):
    plt.plot(xaxs, xu[i-1, :], color="0.8")
plt.plot(xaxs, true_x, "g-", label="true hidden state")
estimate_x = np.zeros([T, 1])
estimate_x = xu.mean(axis=0)
plt.plot(xaxs, estimate_x, "r--", label="estimation")
plt.title("PRC RSMC")
