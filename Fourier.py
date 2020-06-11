import numpy as np
import Common
import unittest
from scipy.signal.windows import chebwin
from time import time_ns


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

    def __init__(self, fs, N, history_len=1, strict=False, freqs=None):
        """:arg freqs Frequencies used in transform"""

        super().__init__(fs, N, history_len=history_len, strict=strict, dtype=np.complex)

        if freqs is None or len(freqs) == 0:
            self.freqs = np.arange(start=0, stop=10000, step=0.5)
        else:
            if not isinstance(freqs, np.ndarray):
                self.freqs = np.ndarray([len(freqs)])
                self.freqs[:] = freqs
            else:
                self.freqs = freqs

        self.A = np.ndarray([len(self.freqs), N], dtype=np.complex)
        self.make_matrix()
        self.reshape_history(len(self.freqs))

    def make_matrix(self):
        """"""

        f = self.freqs / self.fs
        n = np.arange(0, self.N)
        for k in range(len(self.freqs)):
            self.A[k, :] = np.exp(-1j*2*np.pi*f[k]*n)
        self.A /= np.sqrt(self.N)

    def reshape(self, N):
        """Overriding because of constant frequency vector length. Change only number of samples"""

        old = self.N
        self.N = N
        try:
            self._adjust_offset()
        except RuntimeError:
            self.N = old
            raise
        self.A = np.ndarray([len(self.freqs), self.N], dtype=self.A.dtype)
        self.make_matrix()
        self.reshape_history(N)
        self.window = chebwin(self.N, self.attenuation, False)

    def make_freqs_vec(self):
        """
        In case of DtFT frequencies are not dependent on number of samples,
        but stay dependent on sampling frequency. As usually analysis matrix needs to be reshaped only
        when N is changed and frequencies vector when N OR fs is changed. As here frequencies are not
        changed this function is used to indicate change of fs what implies the need to recalculate matrix A
        """
        # For compatibility
        if self.freqs is None:
            self.freqs = np.ndarray([1])

        self.make_matrix()


class FFT(Common.BaseAS):
    """Fast Fourier Transform"""

    def analyse(self, x=None):
        """Multiply signal x with class's orthogonal transform matrix A
           :arg x Signal vector, if None x is N samples from signal starting from self.position in signal
                  If there's less than N samples in signal rest is 0.
                  If whole signal has been processed then zeros vector is returned
           :return A*x' """

#       If signal data is over return zeros
        if self.finished:
            res = np.ones([1, self.N])
            self.saveA(res)
            self.saveT(0)
            return res

        if x is None:
            # Largest index for signal (slice < , ) indexing)
            end = min(self.position + self.N, self.siglen)
            #           Length of available signal
            diff = end - self.position
            #           If available signal is shorter than N samples
            if diff < self.N - 1:
                # Zero initialization
                x = np.zeros([1, self.N])
                #               Copy available part if there's some
                if self.position < self.siglen:
                    x[0, 0:diff] = self.signal[0, self.position:end]
                self.finished = True
            #           If it's possible use only signal
            else:
                x = self.signal[0, self.position:end]
            #           Update processing position
            self.position += self.offset

        start = time_ns()
        if self.use_window and self.window is not None:
            x = np.multiply(x, self.window)
        X = np.fft.fft(np.transpose(x))
        end = time_ns()
        self.saveA(X)
        self.saveT(end - start)
        return X

    def make_matrix(self):
        pass


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
