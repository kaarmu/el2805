import numpy as np

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


class EpsilonGreedy:

    def __init__(self, eps_min, eps_max, z, decay_method='linear'):
        assert hasattr(self, f'decay_{decay_method}'), 'Invalid decay method'
        self._method = getattr(self, f'decay_{decay_method}')
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.z = z

    def __call__(self, k):
        return self._method(k)

    def decay_linear(self, k):
        span = self.eps_max - self.eps_min
        rate = (k-1)/(self.z - 1)
        return max(self.eps_min, self.eps_max - span*rate)

