import numpy as np
import Common
import unittest


class DFT(Common.BaseAS):
    """Discrete Fourier Transform"""

    def __init__(self, fs, N, history_len=1, strict=False):

        super().__init__(fs, N, history_len=history_len, strict=strict,  dtype=np.complex)
        self.make_matrix()

    def make_matrix(self):

        n = np.arange(self.N) / self.N
        for k in range(self.N):
            self.A[k, :] = np.exp(-1j*k*2*np.pi*n)
        self.A /= np.sqrt(self.N)


class DtFT(Common.BaseAS):
    """Discrete time Fourier Transform"""

    def __init__(self, fs, N, freqs, history_len=1, strict=False):
        """:arg freqs Frequencies used in transform"""

        super().__init__(fs, N, history_len=history_len, strict=strict,  dtype=np.complex)

        if not isinstance(freqs, np.ndarray):
            self.freqs = np.ndarray([len(freqs)])
            self.freqs[:] = freqs
        else:
            self.freqs = freqs
        self.A = np.ndarray([len(freqs), N], dtype=np.complex)
        self.make_matrix()
        self.reshape_history(len(self.freqs))

    def make_matrix(self):
        """"""

        f = self.freqs / self.fs
        n = np.arange(0, self.N)
        for k in range(len(self.freqs)):
            self.A[k, :] = np.exp(-1j*2*np.pi*f[k]*n)
        self.A /= np.sqrt(self.N)


class FFT(Common.BaseAS):
    """Fast Fourier Transform"""



class Test(unittest.TestCase):

    def test_dft(self):
        fs = 1000
        N = 100
        dft = DFT(fs, N)
        sig = np.cos(2*np.pi*20*np.linspace(0, (N-1)/fs, N))
        dft.analyse(sig)
        self.assertEqual(np.argmax(dft.getHistoryA()[0:int(N/2)]), 2)

    def test_dtft(self):
        fs = 1000
        N = 100
        dtft = DtFT(fs, N, [19, 20, 20.5, 21, 22])
        sig = np.cos(2 * np.pi * 20.5 * np.linspace(0, (N-1)/fs, N))
        dtft.analyse(sig)
        self.assertEqual(np.argmax(dtft.getHistoryA()), 2)


if __name__ == "__main__":
    unittest.main()
