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

        self.transform = transform

        if transform == "Hadamard":
            res = np.log2(N)
            if res - np.round(res) != 0:
                self.reshape(np.power(2, np.ceil(res)))

        self.make_matrix()
        self.S = np.transpose(self.A)

    def make_matrix(self):
        """Creates transformation matrix based on parameters set in class"""

        n = np.arange(self.N)

#       Calculations
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
            # Length of bit representation
            M = int(np.ceil(np.log2(self.N)))
            f = np.ndarray([self.N, self.N])
#           Saving results to reduce computational cost
            ns = np.ndarray([self.N, M])
            for k in range(self.N):
                if k == 0:
                    kb = np.zeros([1, M])
                else:
                    kb[0, :] = ns[k, :]
                for n in range(self.N):
                    # First time calculation and result saving
                    if k == 0:
                        # Bit representation of n as array of integers
                        nb = np.zeros([1, M])
                        nr = bin(n)[2:]
                        nb[0, -len(nr):] = [int(x) for x in nr]
                        ns[n, :] = nb
                    # Value
                    f[k, n] = sum(np.multiply(kb[0, :], ns[n, :]))
                    self.A[k, n] = np.power(-1, f[k, n])
            self.A /= np.power(2, M/2)

    def reshape(self, N):
        """If Hadamard transform chosen then make sure size is power of 2"""

        if self.transform == "Hadamard":
            N = np.power(2, np.ceil(np.log2(N)))
        super().reshape(int(N))

    def make_freqs_vec(self):

        super(OT, self).make_freqs_vec()
        self.freqs /= 2


if __name__ == "__main__":
    f = OT(100, 8, "Hadamard")
    print(f.A)
