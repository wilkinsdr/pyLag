import numpy as np

def powerlaw(x, norm=1, slope=1):
    return (norm / x[0] ** -slope) * x ** -slope


def powerlaw2(x, norm=1., slope1=1., xbreak=10., slope2=1.):
    func = np.zeros(x.shape)
    func[x <= xbreak] = (norm / x[0] ** -slope1) * x[x <= xbreak] ** -slope1
    func[x > xbreak] = (norm / x[0] ** -slope1) * xbreak ** (slope2 - slope1) * x[x > xbreak] ** -slope2
    return func


def powerlaw3(x, norm=1., slope1=1., xbreak1=10., slope2=1., xbreak2=20., slope3=1.):
    func = np.zeros(x.shape)
    func[x <= xbreak] = (norm / x[0] ** -slope1) * x[x <= xbreak] ** -slope1
    func[x > xbreak] = (norm / x[0] ** -slope1) * xbreak ** (slope2 - slope1) * x[x > xbreak] ** -slope2
    return func


def gaussian(x, norm=1., mu=0., sigma=1.):
   return norm * np.exp((-(x - mu)**2) / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
