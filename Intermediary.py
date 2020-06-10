import OT
import Fourier
import tkinter as tk
import re
import numpy as np
from time import time
import threading
import pyqtgraph as qg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
from PySide2.QtCore import QObject, SIGNAL
import sys
from time import sleep
from tkinter.messagebox import showinfo, showerror
from os.path import exists


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
        self.scale = False
        self.logx = True
        self.logy = True
        self.cv_play = threading.Condition()
        self.cv_sound = threading.Condition()
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
            showinfo("", "Choose transform type first")
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
            try:
                self.transform.reshape(N)
                self.T = self.transform.offset_ms
            except RuntimeError:
                showinfo("Wrong value", "No change made. Value too small")
        self.dispN()
        self.freqs = self.transform.freqs
#       Check if custom attenuation value is given
        self.attenuation_change(None)
#       Check if file was given before
        pth = self.builder.get_object("Path").cget("path")
        if exists(pth):
            self.file_selected(None, pth)

    def file_selected(self, event, pth=None):
        """"""
#       Check if transform class exists. If not prompt and return
        if self.transform is None:
            showinfo("", "Choose transform")
            return
#       Check extension and decide on read type
        if pth is None:
            pth = event.widget.cget("path")
        r = re.compile(".wav$", re.IGNORECASE)
        numeric = False
#       If it's wave file audio read is performed otherwise file is treated as numeric
        if r.search(pth) is None:
            numeric = True
        if numeric:
            # Getting fs value
            try:
                fs = int(self.builder.get_object("SNumericFs").get())
                if fs <= 0:
                    fs = 1000
            except ValueError:
                showerror("Parse error", "Wrong format of sampling frequency for numeric files\nDefault 1000, might"
                                         " be changed by setting value in box and confirming with enter")
                fs = 1000
            self.transform.read_numeric(fs, path=pth)
            self.dispFs()
        else:
            self.transform.read_audio(pth)
#       Time of periodic graph refresh for live audio analysis
        self.T = self.transform.offset_ms
#       Assuming 15ms for drawing in case of constantly operating
        if self.const and self.T < 0.015:
            # Count required windows size
            cnt = int(np.floor(0.015 * self.transform.fs))
#           Reshape to ~20ms window if is too small
            if cnt > self.transform.N:
                self.transform.reshape(int(np.floor(cnt * 4 / 3)))
#           Try setting minimal refresh time
            self.transform.set_offset_ms(15)
            self.T = 0.015
            self.dispN()
#       For fluent display
        elif self.const and self.T > 0.040:
            self.transform.set_offset_ms(40)
            self.T = 0.040
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
        self.cv_sound.acquire()
        self.cv_sound.notify()
        self.cv_sound.release()

    def pause(self, event):
        """Stops dispatching thread"""

        self.play = False

    def log_toggleX(self, event):
        """Change use of logarithmic scale indicator"""

        if self.builder.get_variable("VarLogX").get() == 0:
            self.logx = True
        else:
            self.logx = False

        Plotter.log_mode(self.logx, self.logy)

    def log_toggleY(self, event):
        """Change use of logarithmic scale indicator"""

        if self.builder.get_variable("VarLogY").get() == 0:
            self.logy = True
            self.builder.get_object("CBScale").deselect()
            self.scale = False
        else:
            self.logy = False
            self.builder.get_object("CBScale").select()
            self.scale = True
        Plotter.log_mode(self.logx, self.logy)

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
            self.singer = ThreadE(target=self.sing, name="Music playing thread")
            self.singer.start()
        else:
            if self.singer is not None:
                try:
                    self.singer.terminate()
                except RuntimeError:
                    pass
            self.sound = False

    def window_toggle(self, event):

        if self.transform is None:
            return
        if self.builder.get_variable("VarUse").get() == 0:
            self.transform.use_window = True
        else:
            self.transform.use_window = False

    def trim_toggle(self, event):

        if self.transform is None:
            return
        if self.builder.get_variable("VarTrim").get() == 0:
            self.transform.set_trim(True)
        else:
            self.transform.set_trim(False)

    def reset(self, event):

        self.play = False
        self.transform = None
        self.type = -1
        self.volume = 0.2
        self.T = 1
        self.sound = False
        if self.singer is not None:
            try:
                self.singer.terminate()
            except RuntimeError:
                pass
        self.singer = None
        self.const = True
        self.freqs = None
        self.scale = False
        self.logx = True
        self.logy = True
        self.ui.builder.get_object("Volume").set(0.2)
        self.ui.builder.get_object("CBSound").deselect()
        self.ui.builder.get_object("CBLogX").select()
        self.ui.builder.get_object("CBLogY").select()
        self.ui.builder.get_object("CBScale").deselect()
        self.ui.builder.get_object("CBTrim").deselect()
        self.ui.builder.get_object("CBConst").select()
        self.ui.builder.get_object("CBUse").select()
        self.builder.get_object("Path").configure(path="")
        self.builder.get_object("SN").configure(values=1000)
        self.dispAtt(50)
        Plotter.reset()

    def attenuation_change(self, event):

        if self.transform is None:
            return
        try:
            self.transform.set_attenuation(int(self.builder.get_object("EAttenuation").get()))
        except ValueError:
            showinfo("Wrong attenuation value, setting default 50")
            self.transform.set_attenuation(50)
            self.dispAtt(50)

    def dispN(self):
        """Corrects displayed number of samples per operation. Used when calculations
        force diffrent N than set by user"""

        SN = self.builder.get_object("SN")
#       Clear
        SN.delete(0, len(SN.get()))
#       Set new value
        SN.insert(0, str(self.transform.N))

    def dispAtt(self, val):

        eatt = self.builder.get_object("EAttenuation")
        eatt.delete(0, tk.END)
        eatt.insert(0, str(val))

    def dispFs(self):

        self.builder.get_object("SNumericFs")
        spin = self.builder.get_object("SNumericFs")
#       Clear
        spin.delete(0, tk.END)
#       Set new value
        spin.insert(0, str(self.transform.N))

    def next(self, event):

        if self.const:
            return
        self.cv_play.acquire()
        self.cv_play.notify()
        self.cv_play.release()

    def sing(self):
        """Plays N music samples starting from current position in signal processing"""

        self.cv_sound.acquire()
        self.cv_sound.wait()
        self.cv_sound.release()
        if self.const:
            self.transform.play(self.volume, False, True, self.transform.position)


class Plotter(QMainWindow):

    WindowPlot = None
    WidgetPlot = None
    data = None

    def __init__(self, ui, condition):

        super().__init__()

        Plotter.WindowPlot = qg.GraphicsWindow()
        Plotter.WidgetPlot = self.WindowPlot.addPlot()
        Plotter.start = time()
        self.w = Wrapper()
        self.dispatcher = Dispatcher(ui, condition, self, self.w)
        self.dispatcher.start()
        Plotter.data = None
        Plotter.i = 0
        Plotter.log_mode(True, True)
        Plotter.AT = Intermediary.instance.ui.builder.get_object("LAudioTimeVal")

    @staticmethod
    @pyqtSlot(np.ndarray, name="plot", result="void")
    def plot():
        """Function regularly called by matplotlib.animation"""

        print("p: " + str(Plotter.i))
        Plotter.i += 1
        print(time() - Plotter.start)
        Plotter.start = time()
        Plotter.WidgetPlot.clear()
        if Intermediary.instance.logy:
            Plotter.data = 20*np.log10(np.abs(Plotter.data))
            np.nan_to_num(Plotter.data, False, nan=0.0)
        elif Intermediary.instance.scale:
            M = np.max(np.abs(Plotter.data))
            Plotter.WidgetPlot.setLimits(yMin=-M, yMax=M, minYRange=2*M)
        if Intermediary.instance.logx:
            freqs = np.log10(Intermediary.instance.transform.freqs)
            freqs[0] = 0  # NaN
        else:
            freqs = Intermediary.instance.transform.freqs
        Plotter.WidgetPlot.plot(freqs, Plotter.data)
        Plotter.WidgetPlot.show()
        # Plotter.start = start
        #print(time() - Plotter.start)

        Plotter.AT.configure(text=str(Intermediary.instance.transform.position_time()))

    @staticmethod
    def log_mode(ax:bool=None, ay:bool=None):

        if Plotter.WidgetPlot is not None:
            if ax is not None:
                Plotter.WidgetPlot.getAxis("bottom").setLogMode(ax)
            if ay is not None:
                if ay:
                    Plotter.WidgetPlot.enableAutoRange(qg.ViewBox.YAxis, False)
                    Plotter.WidgetPlot.getAxis("left").setTickSpacing()
                    Plotter.WidgetPlot.setLimits(yMin=0, minYRange=120, yMax=120)
                else:
                    Plotter.WidgetPlot.getAxis("left").setTickSpacing(levels=[])
                    Plotter.WidgetPlot.enableAutoRange(qg.ViewBox.YAxis, True)

    @staticmethod
    def reset():

        Plotter.AT.configure(text=str(0))


class Dispatcher(QThread):

    def __init__(self, ui, condition, plotter, wrapper):

        super().__init__()
        self.ui = ui
        self.intermediary = self.ui.intermediary
        self.cv = condition
        self.w = wrapper
        self.plotter = plotter
        self.strt = time()
        self.i = 0

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
                        time_left = self.intermediary.T - time() + start
                        if time_left > 0.005:
                            sleep(time_left - 0.002)
                        else:
                            pass
                    start = time()
                    try:
                        self.forward()
                    except RuntimeError as e:
                        tk.messagebox.showerror("", e)
                        self.intermediary.play = False
                        break
            else:
                self.forward()

    def forward(self):
        """Controls plotting results and operation of buffering"""

        print("f " + str(self.i))
        self.i += 1
        self.intermediary.transform.analyse()
        Plotter.data = self.intermediary.transform.getHistoryA()
        if Plotter.data.dtype == np.complex:
            Plotter.data = np.abs(Plotter.data)
        self.w.emit(SIGNAL("plot()"))
        if self.intermediary.transform.finished:
            self.intermediary.play = False
            showinfo("", "End of file")
        self.strt = time()


class Worker:

    def __init__(self, ui, condition):

        self.app = QApplication([])
        self.plotter = Plotter(ui, condition)

        self.run()

    def run(self):

        sys.exit(self.app.exec_())


class Wrapper(QObject):

    signal = pyqtSignal(np.ndarray)

    def __init__(self):

        QObject.__init__(self)
        self.connect(self, SIGNAL("plot()"), Plotter.plot)


class ThreadE(threading.Thread):

    def terminate(self):

        raise RuntimeError