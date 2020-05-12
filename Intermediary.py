import OT
import Fourier
import tkinter as tk
import re
import numpy as np
from time import time
import threading
import pyqtgraph as qg
from numpy.random import randn
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, QThreadPool, pyqtSlot, pyqtSignal
from PySide2.QtCore import Signal, Slot, QObject, SIGNAL, SLOT


class Intermediary:
    """Class provides UI with data from backend and forwards user input to backend"""

    instance = None

    def __init__(self, ui):

        if Intermediary.instance is not None:
            raise RuntimeError
        else:
            Intermediary.instance = self

        self.transform = None
        self.ui = ui
        self.builder = ui.builder
        self.Tlist = ["DCT-I", "DCT-II", "DCT-III", "Hadamard", "DFT", "DtFT", "FFT"]
        self.Clist = [OT.OT, OT.OT, OT.OT, OT.OT, Fourier.DFT, Fourier.DtFT, Fourier.FFT]
        self.Tdict = {(k, v) for k, v in zip(self.Tlist, self.Clist)}
        self.type = -1
        self.volume = 0.2
        self.T = 1
        self.sound = False
        self.play = False
        self.singer = None
        self.const = True
        self.freqs = None
        self.scale = True
        self.logx = True
        self.cv_play = threading.Condition()
        self.dispatcher = threading.Thread(target=Dispatcher, args=[self.ui, self.cv_play])

    def init_transform(self, strr):
        pass

    def type_confirmed(self, event):
        """Construct proper transform class accordingly to selection
           from listbox after confirmation button was pressed"""

#       Read selection
        idx = self.builder.get_object("TransformType").curselection()
        if len(idx) > 1:
            raise RuntimeError("Too many choices from listbox")
        if len(idx) == 0:
            tk.messagebox.showinfo("", "Choose transform type first")
            return
        idx = idx[0]
#       Check if change was made
        N = int(self.builder.get_object("SN").get())
        if self.type == idx and N == self.transform.N:
            return
#       If type changed update type info
        if self.type != idx:
            self.type = idx
#           Check module and initialize transform class
            if self.Clist[idx].__module__ == "Fourier":
                self.transform = self.Clist[idx](100, N, 50, False)
            else:
                self.transform = self.Clist[idx](100, N, self.Tlist[idx], 50, False)
#       If N changed for the same transform type
        elif self.transform.N != N:
            self.transform.reshape(N)
        self.dispN()
        self.freqs = self.transform.freqs

    def file_selected(self, event):
        """"""
#       Check if transform class exists. If not prompt and return
        if self.transform is None:
            tk.messagebox.showinfo("", "Choose transform type first")
            return
#       Check extension and read audio file
#       self.builder.get_object("Path")
        pth = event.widget.cget("path")
        r = re.compile(".wav$", re.IGNORECASE)
        if r.search(pth) is None:
            tk.messagebox.showerror("", "Only .wav files are allowed")
            return
        self.transform.read_audio(pth)
#       Time of periodic graph refresh for live audio analysis
        self.T = round(self.transform.N / self.transform.fs)
#       Assuming 25ms for drawing in case of constantly operating
        if self.const and self.T < 0.025:
            self.transform.reshape(int(np.floor(0.025 * self.transform.fs)))
            self.T = np.around(self.transform.N / self.transform.fs, 3)
            self.dispN()
#       Initialize buffers if needed else change data
        self.freqs = self.transform.freqs

    def start(self, event):
        """Start dispatching thread"""

        if self.transform is None or self.play:
            return
        self.play = True
        self.cv_play.acquire()
        self.cv_play.notify()
        self.cv_play.release()

    def pause(self, event):
        """Stops dispatching thread"""

        self.play = False

    def log_toggle(self, event):
        """Change use of logarithmic scale indicator"""

        if self.builder.get_variable("VarLog").get() == 0:
            self.logx = True
        else:
            self.logx = False

    def scale_toggle(self, event):
        """Change use of scaling indicator"""

        if self.builder.get_variable("VarScale").get() == 0:
            self.scale =  True
        else:
            self.scale = False

    def const_toggle(self, event):
        """Switch indicator between constant play or stepped"""

        if self.builder.get_variable("VarConst").get() == 0:
            self.const = True
        else:
            self.const = False

    def sound_toggle(self, event):
        """Start/stop music playing thread"""

        if self.builder.get_variable("VarSound").get() == 0:
            self.sound = True
            self.singer = threading.Thread(target=self.sing, name="Music playing thread")
            self.singer.start()
        else:
            self.sound = False

    def dispN(self):
        """Corrects displayed number of samples per operation. Used when calculations
        force diffrent N than set by user"""

        SN = self.builder.get_object("SN")
#       Clear
        SN.delete(0, len(SN.get()))
#       Set new value
        SN.insert(0, str(self.transform.N))

    def next(self, event):

        if self.const:
            return
        self.cv_play.acquire()
        self.cv_play.notify()
        self.cv_play.release()

    def sing(self):
        """Plays N music samples starting from current position in signal processing"""

        #self.transform.play(self.volume, False, True)
        while self.sound:
            self.cv_play.acquire()
            self.cv_play.wait()
            self.cv_play.release()
            self.transform.play(self.volume, False)


class Plotter(QMainWindow):

    def __init__(self):

        super().__init__()

        self.WidgetPlot = qg.PlotWidget()
        self.setCentralWidget(self.WidgetPlot)
        self.show()
        self.WidgetPlot.plot([1, 2], [1, 1])

        # self.i = 0
        # self.AT = self.ui.builder.get_object("EAudioTime")

    def plot(self, data):
        """Function regularly called by matplotlib.animation"""

        self.WidgetPlot.clear()
        self.WidgetPlot.plot(Intermediary.instance.transform.freqs, data)

# #       Clear
#         self.AT.delete(0, len(self.AT.get()))
# #       Set new value
#         self.AT.insert(0, str(self.i * self.intermediary.T))
#         self.i += 1


class Dispatcher(QThread):

    def __init__(self, ui, condition, plotter):

        super().__init__()
        self.ui = ui
        self.intermediary = self.ui.intermediary
        self.cv = condition
        self.plot = plotter.plot

    def run(self):
        """If program runs in constant audio playing mode than function calls every period
        of refreshing forward function"""

        while True:
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
            start = time()
            if self.intermediary.const:
                while self.intermediary.play:
                    while time() - start < self.intermediary.T:
                        pass
                    try:
                        self.forward()
                    except RuntimeError as e:
                        tk.messagebox.showerror("", e)
                        self.intermediary.play = False
                        break
                    start = time()
            else:
                self.forward()

    def forward(self):
        """Controls plotting results and operation of buffering"""

        #self.intermediary.transform.analyse()
        self.plot(randn(len(self.intermediary.transform.freqs)))
        #self.plot(self.intermediary.transform.getHistoryA())


class Worker:

    def __init__(self, ui, condition):

        self.app = QApplication([])
        self.plotter = Plotter()
        self.dispatcher = Dispatcher(ui, condition, self.plotter)
        self.dispatcher.start()
        self.run()

    def run(self):

        self.app.exec_()
