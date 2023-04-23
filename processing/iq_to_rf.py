import numpy as np

def iq_to_rf(iq, w0):
    return np.real(iq * np.exp(1j*w0))