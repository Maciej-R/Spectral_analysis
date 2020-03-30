import numpy as np
import Common


class OT(Common.BaseAS):
    """Orthogonal transforms"""

    def __init__(self, fs=8000, N=250, transform="DCT-II", history_len=1, strict=False):
        """:arg fs Sampling rate for processed signal
           :arg N Number of samples per operation
           :arg transform Type of orthogonal transformation used in class,
                 available are DCT-I, DCT-II, DCT-III, Hadamard - N must be 2^"""

        super().__init__(fs, N, history_len, strict)

        if transform == "Hadamard":
            res = np.log2(N)
            if res - np.round(res) != 0:
                print("Wrong N for Hadamard transform")

        self.transform = transform

        self.make_matrix()
        self.S = np.transpose(self.A)

    def make_matrix(self):
        """Creates transformation matrix based on parameters set in class"""

        n = np.arange(self.N)

        if self.transform == "DCT-I":
            for k in range(0, self.N):
                self.A[k, :] = (np.cos(np.pi*k*n/self.N))
            self.A /= np.sqrt(self.N)
            self.A *= np.sqrt(2)
            self.A[0, :] /= np.sqrt(2)
            self.A[:, 0] /= np.sqrt(2)

        elif self.transform == "DCT-II":
            for k in range(0, self.N):
                self.A[k, :] = (np.cos(np.pi*k*(n + 0.5)/self.N))
            self.A /= np.sqrt(self.N)
            self.A[1:self.N-1, :] *= np.sqrt(2)

        elif self.transform == "DCT-III":
            for k in range(0, self.N):
                self.A[k, :] = (np.cos(np.pi*(k + 0.5)*n/self.N))
            self.A /= np.sqrt(self.N)
            self.A[:, 1:self.N - 1] *= np.sqrt(2)

        elif self.transform == "Hadamard":
            M = int(np.log2(self.N))

            f = np.ndarray([self.N, self.N])
            ns = np.ndarray([self.N, M])
            for k in range(self.N):
                kb = np.zeros([1, M])
                kr = bin(k)[2:]
                kb[0, -len(kr):] = [int(x) for x in kr]
                #ks = [np.power(2, i)*kb[i] for i in range(min(M, len(kb)))]
                #ks[len(ks):M-1] = np.zeros([1, M-1-len(ks)]);
                for n in range(self.N):
                    if k == 0:
                        nb = np.zeros([1, M])
                        nr = bin(n)[2:]
                        nb[0, -len(nr):] = [int(x) for x in nr]
                        ns[n, :] = nb
                        #ns[n] = [np.power(2, i)*nb[i] for i in range(min(M, len(nb)))]
                        #ns[len(ns):M - 1] = np.zeros([1, M - 1 - len(ns)]);
                    f[k, n] = sum(np.multiply(kb[0, :], ns[n, :]))
                    self.A[k, n] = np.power(-1, f[k, n])
            self.A /= np.sqrt(self.N)

    def read_audio(self, path):

        super().read_audio(path)
        self.freqs /= 2

    def read_numeric(self, data, fs, *, dtype=None, fill=False):
        """Overriding Signal function for frequency vector correction (compared to DFT it's a half)"""

        super().read_numeric(data, fs,  dtype=None, fill=False)
        self.freqs /= 2

    def reshape(self, N):
        """"""
        super().reshape(N)
        self.freqs /= 2


if __name__ == "__main__":
    f = OT(100, 8, "Hadamard")
    print(f.A)
