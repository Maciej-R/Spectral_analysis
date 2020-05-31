import simpleaudio as sa
import numpy as np
import threading
import unittest
import struct
import abc
from enum import Enum
from time import time_ns
import wave
import copy
import scipy.signal as sp


class Base:
    """Class with basing file / transform parameters"""

    def __init__(self, fs, N, dtype=None):
        """:arg N Number of samples per operation
           :arg fs Sampling Frequency"""
        self.N = N
        self.fs = fs
        self.freqs = None
        self.position = 0
        self.make_freqs_vec()
        self.A = np.ndarray([self.N, self.N], dtype=dtype)
        self.S = np.ndarray([self.N, self.N], dtype=dtype)
        self.attenuation = 50
        self.window = sp.windows.chebwin(self.N, self.attenuation, False)
        self.offset = N/2
        self.offset_ms = np.around(self.offset / self.fs, 3)

    def setFs(self, fs):
        """Set sampling rate"""
        self.fs = fs

    def make_freqs_vec(self):
        """Calculate vector of frequencies present in transform (orthogonal, DFT)"""
        self.freqs = np.arange(start=0, stop=self.N) / self.N * self.fs

    def set_attenuation(self, att):
        """Set desired window attenution"""

        self.attenuation = att
        self.window = sp.windows.chebwin(self.N, self.attenuation, False)

    def _adjust_offset(self):
        """
        If parameter offset_ms is set than adjustments to match current N and fs are made.
        Done automatically when needed
        """

        if self.offset_ms is None:
            self.offset = int(self.N/2)
        else:
            tmp_offset = np.round(self.offset_ms * self.fs)
            if tmp_offset > self.N:
                raise RuntimeError
            else:
                self.offset = int(tmp_offset)

    def set_offset_ms(self, offset):
        """
        Adjust position change to match given time restrictions.
        Makes shifts independent from window length and sampling frequency allowing display to stay smooth
        even if fs is low causing long refresh times with given window length.
        Default offset is N/2 (so T = fs/N*2 where N-window length and fs-sampling frequency).
        Doesn't start any internal timers. All actions should be triggered from outside.
        :arg offset Time of analysis window shift in milliseconds
        """

        self.offset_ms = offset/1000
        self._adjust_offset()


class AnalysisResultSaver(Base):
    """Class allows to save history of transforms and to read it"""

    class SaveType(Enum):

        analysis = 1
        synthesis = 2
        time = 4

    def __init__(self, fs=8000, N=250, history_len=1, strict=False, dtype=None, shape=None):
        """:arg fs Sampling rate
           :arg N Number of samples per analysis
           :arg history_len Number of records in history
           :arg strict Strict mode checks whether all history types (analysis, synthesis and time) are up to date
           :arg dtype Data type for analysis and synthesis history"""

        super().__init__(fs, N, dtype)

#       Data structures for history
        self.analysis = np.ndarray([history_len, N], dtype=dtype)
        self.synthesis = np.ndarray([history_len, N], dtype=dtype)
        self.time = np.ndarray([history_len, 1])
#       Indexes, parameter, lock
        self.a_row = 0
        self.s_row = 0
        self.t_row = 0
        self.strict = strict
        self.dtype = dtype
        self.history_len = history_len
        self.mutex = threading.Lock()

    def saveA(self, X):
        """Short for save(X, SaveType.analysis)"""

        self.save(X, self.SaveType.analysis)

    def saveS(self, X):
        """Short for save(X, SaveType.synthesis)"""

        self.save(X, self.SaveType.synthesis)

    def saveT(self, X):
        """Short for save(X, SaveType.time)"""

        self.save(X, self.SaveType.time)

    def save(self, X, t):
        """Saves are synchronized, thread safe
           :arg X Vector of self.N values to save or int in case t == SaveType.time
           :arg t Type of result to save (analysis, synthesis, time, from SaveType enum)
           :raises RuntimeError when strict is set and difference between any rows in history is bigger than 1"""

#       No history being saved
        if self.history_len == 0:
            return

        if isinstance(X, np.ndarray) and len(X.shape) != 1:
            X = X.reshape([len(X)])

#       Save date, update row index
        try:

            self.mutex.acquire()
            if t == self.SaveType.analysis:
                self.analysis[self.a_row, :] = X
                self.a_row += 1
                self.a_row %= self.history_len
            elif t == self.SaveType.synthesis:
                self.synthesis[self.s_row, :] = X
                self.s_row += 1
                self.s_row %= self.history_len
            elif t == self.SaveType.time:
                self.time[self.t_row, 0] = X
                self.t_row += 1
                self.t_row %= self.history_len

        finally:

            if self.mutex.locked():
                self.mutex.release()

#       All histories have to be updated coherently is strict is enabled
        if self.strict:

            ts = abs(self.t_row - self.s_row) % self.history_len
            ta = abs(self.t_row - self.a_row) % self.history_len
            sa = abs(self.s_row - self.a_row) % self.history_len

            if ts > 1 or ta > 1 or sa > 1:
                raise RuntimeError("History not synchronized")

    def getHistoryA(self, back=0, synchronize=False):
        """Short for getHistory(back, SaveType.analysis)"""

        return self.getHistory(self.SaveType.analysis, back, synchronize)

    def getHistoryS(self, back=0, synchronize=False):
        """Short for getHistory(back, SaveType.synthesis)"""

        return self.getHistory(self.SaveType.synthesis, back, synchronize)

    def getHistoryT(self, back=0, synchronize=False):
        """Short for getHistory(back, SaveType.time)"""

        return self.getHistory(self.SaveType.time, back, synchronize)

    def getHistory(self, t, back=0, synchronize=False):
        """Get record from history of analysis
           :arg back Value from 0 to history_len-1 where 0 is latest and history_len-1 is oldest record
           :arg t Type of history to
           :arg synchronize True if mutex should be used (introduces some delay)"""

#       If no history saved
        if self.history_len == 0:
            return None

#       Maximum backward indexing
        if back > self.history_len - 1:
            back = self.history_len - 1

#       Save to proper table
        res = None
        try:

            if synchronize:
                self.mutex.acquire()
            if t == self.SaveType.analysis:
                res = self.analysis[self.a_row - back - 1, :]
            elif t == self.SaveType.synthesis:
                res = self.synthesis[self.s_row - back - 1, :]
            elif t == self.SaveType.time:
                res = self.time[0, self.t_row - back - 1]

        finally:

            if synchronize and self.mutex.locked():
                self.mutex.release()

        return res

    def reshape_history(self, N):
        """When number of samples per analysis changes, update history matrix size, synchronized"""

        try:

            self.mutex.acquire()
            self.analysis = np.ndarray([self.history_len, N], dtype=self.dtype)
            self.synthesis = np.ndarray([self.history_len, N], dtype=self.dtype)
            self.time = np.ndarray([self.history_len, N], dtype=self.dtype)
            self.make_freqs_vec()

        finally:

            if self.mutex.locked():
                self.mutex.release()


class Signal(Base):

    def __init__(self, fs, N, dtype=None):
        super().__init__(fs, N, dtype)
        self.signal = None
        self.audio = None
        self.sample_size = 1
        self.nchannels = 1
        self.siglen = 0
        self.audio_data = False

    def read_audio(self, path):
        """Function imports .wav file
           :arg path Path to file """

#       Open audio file
        reader = wave.open(path, "rb")
#       Set frame rate
        self.setFs(reader.getframerate())
        self.make_freqs_vec()
#       Get recording parameters
        ls = reader.getnframes()
        self.nchannels = reader.getnchannels()
        self.sample_size = reader.getsampwidth()

#       Read and parse signal
        tmp_sig = reader.readframes(ls)
        fmt = {1: "%db", 2: "<%dh", 4: "<%dl"}[self.sample_size] % (ls * self.nchannels)
#       Read original version as numerical array
        parsed = struct.unpack(fmt, tmp_sig)
        self.audio = np.ndarray([1, len(parsed)], dtype=("<i%d" % self.sample_size))
        self.audio[0, :] = parsed

#       If stereo convert to mono
        if self.nchannels == 2:
            self.signal = np.ndarray([1, int(len(self.audio[0, :])/2)], dtype=self.audio.dtype)
            self.signal[0, :] = 0.5 * self.audio[0, ::2] + 0.5 * self.audio[0, 1::2]
        elif self.nchannels == 1:
            self.signal = self.audio

        self.audio_data = True
        self.position = 0
        self.siglen = len(self.signal[0, :])

    def read_numeric(self, data, fs, *, dtype=None, fill=False):
        """Function imports data from numeric vector (data is deep-copied)
           :param dtype Data type to be used, if non given data.dtype is used
           :param fill If true and data length < N than fill rest with zeros (to N)
           :arg data Data source as numpy.ndarray([1, data_length]) or numpy.ndarray([length])
           :arg fs Sampling rate
           :raises RuntimeError When data length < self.N
           :raises TypeError When isinstance(data, np.ndarray) == False"""

        if not isinstance(data, np.ndarray):
            raise TypeError("numpy.ndarray expected")

#       If array is np.ndarray([length]) than reshape to np.ndarray([1, length])
        if len(data.shape) == 1:
            data = data.reshape([1, len(data)])

#       Signal too short with current settings
        if len(data[0, :]) < self.N and not fill:
            raise RuntimeError

#       Zero-filled short signal
        if len(data[0, :]) < self.N and fill:
            self.signal = np.zeros([1, self.N])
            self.signal[0, 0:len(data[0, :])] = copy.deepcopy(data)
            return

#       Normal case
        self.signal = copy.deepcopy(data)
        if dtype is not None and dtype != data.dtype:
            self.signal = np.ndarray.astype(self.signal, dtype)

#       Set class parameters for given signal
        self.setFs(fs)
        self.make_freqs_vec()
        self.nchannels = 1
        self.audio = None
        self.audio_data = False
        self.sample_size = self.signal.dtype.itemsize
        self.position = 0
        self.siglen = len(data)

    def play(self, volume=0.2, pre=True, whole=False, start=0):
        """Function plays signal as wave sound file
           :arg pre Defines if sound is being played before of after doing analysis. In first case sound is
                    played from current position of processing otherwise it's N samples back
            :arg whole Whether to play whole signal or N samples
            :arg volume Music amplitude is multiplied by this value
            :arg start Starting position, relevant if whole == True"""

        if whole:
            sa.play_buffer(np.asarray(self.signal[0, start:] * volume, dtype=self.signal.dtype),
                           1, self.sample_size, self.fs)
            return

        if pre:
            sa.play_buffer(np.asarray(self.signal[0, self.position-self.N:self.position] * volume, self.signal.dtype),
                           1, self.sample_size, self.fs)
        else:
            sa.play_buffer(np.asarray(self.signal[0, self.position:self.position+self.N], self.signal.dtype),
                           1, self.sample_size, self.fs)


class BaseAS(abc.ABC, AnalysisResultSaver, Signal):
    """Class implementing analyse and synthesize method + abstract make_matrix method"""

    def analyse(self, x=None):
        """Multiply signal x with class's orthogonal transform matrix A
           :arg x Signal vector, if None x is N samples from signal starting from self.position in signal
                  If there's less than N samples in signal rest is 0
           :return A*x' """

        if x is None:
            # Largest index for signal (slice < , ) indexing)
            end = min(self.position+self.N, self.siglen)
#           Length of available signal
            diff = end-self.position
#           If available signal is shorter than N samples
            if diff < self.N - 1:
                #Zero initialization
                x = np.zeros([1, self.N])
#               Copy available part if there's some
                if self.position < self.siglen:
                    x[0, 0:diff] = self.signal[0, self.position:end]
#           If it's possible use only signal
            else:
                x = self.signal[0, self.position:end]
#           Update processing position
            self.position += self.offset

        start = time_ns()
        if self.window is not None:
            x = np.multiply(x, self.window)
        X = np.dot(self.A, np.transpose(x))
        end = time_ns()
        self.saveA(X)
        self.saveT(end - start)
        return X

    def synthesize(self, X=None):
        """Synthesize signal from transform values
           :arg X Transform data from which signal will be synthesized, if None X is taken from history of analysis
           :return S*X """
#       Get X from history if None passed in argument
        if X is None:
            X = self.getHistoryA()
        x = np.dot(self.S, X)
        self.saveS(x)
        return x

    @abc.abstractmethod
    def make_matrix(self):
        """Should be implemented and called in constructor for analyse and synthesize to work
           As result matrix A ought to be filled (S might be transposed in constructor)"""
        pass

    def reshape(self, N):
        """
        Reshape transformation matrix, history structures, window, recalculate values,
        adjust offset (if this fails nothing is changed, error is thrown) see set_offset_ms
        """

        old = self.N
        self.N = N
        try:
            self._adjust_offset()
        except RuntimeError:
            self.N = old
            raise
        self.A = np.ndarray([self.N, self.N], dtype=self.A.dtype)
        self.make_matrix()
        self.reshape_history(N)
        self.window = sp.windows.chebwin(self.N, self.attenuation, False)


class Test(unittest.TestCase):

    def test_numeric(self):
        x = np.ndarray([1, 10])
        x[0, :] = np.arange(10)
        N = 10
        sig = Signal(0, N)
        fs = 100
        sig.read_numeric(x, fs, fill=True)
#       Check read and parameters setting
        self.assertTrue(np.array_equal(x, sig.signal))
        self.assertEqual(sig.fs, fs)
        self.assertEqual(len(x[0, :]), len(sig.signal[0, :]))
#       Check wrong data type
        with self.assertRaisesRegex(TypeError, "numpy.ndarray expected"):
            sig.read_numeric([0, 1, 2], 100)
#       Check reaction to too short signal without filling it
        sig.N = 100
        with self.assertRaises(RuntimeError):
            sig.read_numeric(x, 100, fill=False)
#       Check filling with zeros
        sig.read_numeric(x, fs, fill=True)
        self.assertTrue(np.array_equal(sig.signal[0, 10:100], np.zeros(90)))

    def test_AnalysisResultSaver(self):
        N = 10
#       Check enforcement of strict mode
        obj = AnalysisResultSaver(100, N, 5, True)
        obj.saveT(np.zeros(1))
        self.assertTrue(obj.strict)
        with self.assertRaisesRegex(RuntimeError, "History not synchronized"):
            obj.saveT(2)
#       Check proper saving
        obj = AnalysisResultSaver(100, N, 5, True)
        saved = np.zeros(10)
        saved[3] = 8
        obj.saveA(saved)
        self.assertTrue(np.array_equal(obj.getHistoryA(), saved))

    def test_BaseAS(self):
        class TestBaseAS(BaseAS):
            def make_matrix(self):
                self.A = np.ones([self.N, self.N])
#       Check setting parameters
        fs = 100
        N = 10
        b = TestBaseAS(fs, N)
        self.assertEqual(b.fs, fs)
        self.assertEqual(b.N, N)
        freq = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.assertTrue(np.array_equal(freq, b.freqs))
#       Check reshaping
        N = 100
        b.reshape(N)
        self.assertEqual(b.N, N)
        self.assertEqual(len(b.A[0, :]), N)
#       Check using zeros when signal is over (2) and using positioning (1)
        b.read_numeric(np.arange(N), fs)
        b.analyse()
        self.assertTrue(np.array_equal(b.analyse(), np.zeros([N, 1])))


if __name__ == "__main__":

    #sig = Signal(0, 250)
    #sig.read_audio("The_Rasmus-In_The_Shadows.wav")
    #sa.play_buffer(sig.signal, 1, sig.sample_size, sig.fs)

    unittest.main()

#   sa.play_buffer(sig.mono, 1, sig.signal.itemsize, sig.fs)

#   import matplotlib.pyplot as plt
#   plt.plot(range(100000), oth[0, 330000:430000])
#   #plt.show()

#   import scipy.signal as sc
#   corr = sc.correlate(sig.signal[0, 150000:200000:2], sig.signal[0, 150001:200001:2])
#   plt.plot(range(5000), corr[0:5000])
#   plt.show()

#   writer = wave.open("mono_conversion_1.wav", "wb")
#   writer.setframerate(sig.fs)
#   writer.setnchannels(1)
#   writer.setsampwidth(sig.sample_size)
#   writer.writeframes(oth)
#   writer.close()

#self.mono[0, :] = np.ndarray.astype(0.5 * (np.ndarray.astype(self.signal[0, ::2], dtype=np.int64)
#                                                       + np.ndarray.astype(self.signal[0, 1::2], dtype=np.int64)),
#                                                dtype=self.signal.dtype)