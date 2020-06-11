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
        self.bar = False

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
            try:
                self.transform.read_numeric(fs, path=pth, error=True)
            except RuntimeError:
                showerror("Format error", "Numbers in input file have wrong format")
        else:
            self.transform.read_audio(pth)
        self.dispFs()
#       Time of periodic graph refresh for live audio analysis
        self.T = self.transform.offset_ms
#       Assuming 15ms for drawing in case of constantly operating
        if self.const and self.T < 0.015:
            self.prolong()
#       For fluent display
        elif self.const and self.T > 0.040:
            self.transform.set_offset_ms(40)
            self.T = 0.040
#       Initialize buffers if needed else change data
        self.freqs = self.transform.freqs

    def prolong(self, time=None, multiply=None):
        """
        Prolongs time between next time slots for analysis
        :arg time Value in second for time slot
        :arg mulitply Resulting time slot is current time slot * multiply
        """

        if multiply is not None:
            time = self.T * multiply
#       Default 0.015 time slot
        elif time is None:
            time = 0.015

#       Maximum time allowed
        if time > 0.05:
            raise RuntimeError("Cannot prolong")

#       Try setting given time for current size, if failed resize
        try:
            self.transform.set_offset_ms(time*1000)
            self.T = self.transform.offset_ms
            return
        except:
            pass

#       Count required windows size
        cnt = int(np.floor(time * self.transform.fs))
#       Limit size
        if cnt > 10000:
            raise RuntimeError("Cannot prolong")
#       Reshape to time *4/3 window if is too small
        if cnt > self.transform.N:
            self.transform.reshape(int(np.floor(cnt * 4 / 3)))
#       Try setting minimal refresh time
        self.transform.set_offset_ms(time*1000)
        self.T = time
        self.dispN()

    def start(self, event):
        """Start dispatching thread"""

        if self.transform is None or self.play:
            return
        self.play = True
#       Notify to start music and plotting
        self.cv_sound.acquire()
        self.cv_sound.notify()
        self.cv_sound.release()
        self.cv_play.acquire()
        self.cv_play.notify()
        self.cv_play.release()

    def pause(self, event):
        """Stops dispatching thread"""

        self.play = False
        if self.singer is not None:
            self.singer.stop()

    def log_toggleX(self, event):
        """Change use of logarithmic scale indicator"""

        if self.builder.get_variable("VarLogX").get() == 0:
            self.logx = True
        else:
            self.logx = False

        Plotter.log_mode(self.logx, self.logy)

    def log_toggleY(self, event):
        """Change use of logarithmic scale indicator"""

#       Set parameter and perform default operations on scaling
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

#       Scaling only for linear values
        if self.logy:
            self.builder.get_object("CBScale").select()
            return

#       Plotter y axis ticks need adjustment
        if self.builder.get_variable("VarScale").get() == 0:
            self.scale = True
            Plotter.log_mode(self.logx, self.logy)
        else:
            self.scale = False
            Plotter.log_mode(self.logx, self.logy, True)

    def const_toggle(self, event):
        """Switch indicator between constant play or stepped"""

        if self.builder.get_variable("VarConst").get() == 0:
            self.const = True
        else:
            self.const = False

#       Pause music if it's on and not constant operation mode is chosen
        if self.const and self.singer is not None:

            self.singer.stop()

    def sound_toggle(self, event):
        """Start/stop music playing thread"""

        if self.builder.get_variable("VarSound").get() == 0:
            if not self.const:
                self.builder.get_object("CBConst").deselect()
                return
            self.sound = True
#           Create and start thread if needed
            if self.singer is None:
                self.singer = Singer(self.cv_sound)
                self.singer.start()
#           Notify to start
            if self.play:
                self.cv_sound.acquire()
                self.cv_sound.notify()
                self.cv_sound.release()
#       End thread, sound off
        else:
            if self.singer is not None:
                try:
                    self.singer.terminate()
                except RuntimeError:
                    pass
            self.singer = None
            self.sound = False

    def window_toggle(self, event):
        """Using window in analysis"""

        if self.transform is None:
            return
        if self.builder.get_variable("VarUse").get() == 0:
            self.transform.use_window = True
        else:
            self.transform.use_window = False

    def trim_toggle(self, event):
        """Frequencies limited to half range for DTF, FFT where spectrum is symmetric"""

        if self.transform is None:
            return
        if self.builder.get_variable("VarTrim").get() == 0:
            self.transform.set_trim(True)
        else:
            self.transform.set_trim(False)

    def bar_toggle(self, event):
        """Switch plotting mode"""

        if isinstance(event, bool):
            self.bar = event
            cb = self.builder.get_object("CBBar")
            if event:
                cb.select()
            else:
                cb.deselect()
        elif self.builder.get_variable("VarBar").get() == 0:
            self.bar = True
        else:
            self.bar = False

    def volume_change(self, event):
        """Set new volume value"""

        self.volume = event.widget.get()

    def reset(self, event):
        """Restoring default state"""

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
        self.bar = False
        self.ui.builder.get_object("Volume").set(0.2)
        self.ui.builder.get_object("CBSound").deselect()
        self.ui.builder.get_object("CBLogX").select()
        self.ui.builder.get_object("CBLogY").select()
        self.ui.builder.get_object("CBScale").deselect()
        self.ui.builder.get_object("CBTrim").deselect()
        self.ui.builder.get_object("CBConst").select()
        self.ui.builder.get_object("CBUse").select()
        self.ui.builder.get_object("CBBar").deselect()
        self.builder.get_object("Path").configure(path="")
        self.builder.get_object("SN").configure(values=1000)
        self.dispAtt(50)
        Plotter.reset()

    def attenuation_change(self, event):
        """Sets attenuation value to value from input or default 50"""

        if self.transform is None:
            return
        try:
            self.transform.set_attenuation(int(self.builder.get_object("EAttenuation").get()))
        except ValueError:
            showinfo("Wrong attenuation value, setting default 50")
            self.transform.set_attenuation(50)
#           Display correction, input had wrong value
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
        """Displays current attenuation value for used window"""

        eatt = self.builder.get_object("EAttenuation")
        eatt.delete(0, tk.END)
        eatt.insert(0, str(val))

    def dispFs(self):
        """Displays current sampling frequency set in transform"""

        spin = self.builder.get_object("SNumericFs")
#       Clear
        spin.delete(0, tk.END)
#       Set new value
        spin.insert(0, str(self.transform.fs))

    def next(self, event):
        """Single operation"""

        if self.transform is None:
            return

        if self.const:

            self.builder.get_object("CBConst").deselect()
            self.const_toggle(None)


class Plotter(QMainWindow):
    """Handling graphs window"""

    WindowPlot = None
    WidgetPlot = None
    data = None

    def __init__(self, ui, condition):

        super().__init__()

#       Preparing window
        Plotter.WindowPlot = qg.GraphicsWindow(title="Transform Graphs")
        Plotter.WidgetPlot = self.WindowPlot.addPlot()
        self.w = Wrapper()
        self.dispatcher = Dispatcher(ui, condition, self.w)
        self.dispatcher.start()
#       Parameters setting
        Plotter.data = None
        Plotter.i = 0
        Plotter.log_mode(True, True)
        Plotter.AT = Intermediary.instance.ui.builder.get_object("LAudioTimeVal")
        Plotter.AD = Intermediary.instance.ui.builder.get_object("ADuration")
        Plotter.max = 0
        qg.setConfigOption("foreground", "w")

    @staticmethod
    @pyqtSlot(np.ndarray, name="plot", result="void")
    def plot():
        """Function regularly called by matplotlib.animation"""

        Plotter.i += 1
#       Get rid of previous data
        Plotter.WidgetPlot.clear()
#       Prepare values for logarithmic plot, NaNs cause no display
        if Intermediary.instance.logy:
            Plotter.data = 20*np.log10(np.abs(Plotter.data))
            np.nan_to_num(Plotter.data, False, nan=0.0)
#       If scaling is enabled limits are set accordingly to current max values
        elif Intermediary.instance.scale:
            M = np.max(np.abs(Plotter.data))
            Plotter.WidgetPlot.setLimits(yMin=-M, yMax=M, minYRange=2*M)
#       For linear values adjust limits if values are bigger than 150% of current limits or
#       smaller than 10% of current values
        else:
            M = np.max(np.abs(Plotter.data))
            if M > Plotter.max * 1.5 or M < 0.1 * Plotter.max:
                Plotter.max = M
                Plotter.WidgetPlot.setLimits(yMin=-M, yMax=M, minYRange=2*M)
        if Intermediary.instance.logx:
            freqs = np.log10(Intermediary.instance.transform.freqs)
            freqs[0] = 0  # np.finfo(np.float).eps  # -Inf
        else:
            freqs = Intermediary.instance.transform.freqs
        if not Intermediary.instance.bar:
            Plotter.WidgetPlot.plot(freqs, Plotter.data)
        else:
            Plotter.WidgetPlot.addItem(qg.BarGraphItem(x=freqs, height=Plotter.data, width=(freqs[0]-freqs[1])/2))

#       Display playback time
        Plotter.AT.configure(text=str(np.around(Intermediary.instance.transform.position_time(), 1)))
        Plotter.AD.configure(text=str(np.around(Intermediary.instance.transform.getHistoryT()/1000, 1)))

    @staticmethod
    def log_mode(ax:bool=None, ay:bool=None, ticks=None):
        """Controls changes between linear and logarithmic modes"""

#       Do nothing is initialization hasn't been performed
        if Plotter.WidgetPlot is not None:
            # Set bottom axis mode
            if ax is not None:
                Plotter.WidgetPlot.getAxis("bottom").setLogMode(ax)
            # For consistent display limit range and values in logarithmic mode for left axis
            if ay is not None:
                if ay:
                    Plotter.WidgetPlot.enableAutoRange(qg.ViewBox.YAxis, False)
                    Plotter.WidgetPlot.setLimits(yMin=0, minYRange=120, yMax=120)
                    if ticks is None or ticks:
                        Plotter.WidgetPlot.getAxis("left").setTickSpacing()
                    else:
                        Plotter.WidgetPlot.getAxis("left").setTickSpacing(levels=[])
                # In linear mode scale is not display as values are rapidly changing
                else:
                    Plotter.WidgetPlot.enableAutoRange(qg.ViewBox.YAxis, True)
                    if ticks is None or not ticks:
                        Plotter.WidgetPlot.getAxis("left").setTickSpacing(levels=[])
                    else:
                        Plotter.WidgetPlot.getAxis("left").setTickSpacing()

    @staticmethod
    def reset():
        """Restoring basic state"""

        Plotter.AT.configure(text=str(0))
        Plotter.WidgetPlot.clear()
        Plotter.data = None


class Dispatcher(QThread):
    """Class commissioning jobs"""

    def __init__(self, ui, condition, wrapper):
        """
        :arg wrapper Used to emit signals to Plotter
        :arg cv Condition variable signaling when to start work
        """

        super().__init__()
        self.intermediary = ui.intermediary
        self.cv = condition
        self.w = wrapper
        self.strt = time()
        self.i = 0

    def run(self):
        """If program runs in constant audio playing mode than function calls every period
        of refreshing forward function"""

#       Wait for signal to start work as long as thread is alive
        while True:
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
            start = time()
#           If constant work is requested, do it until play variable signals otherwise
            if self.intermediary.const:
                while self.intermediary.play:
                    # Wait for proper time to continue
                    while time() - start < self.intermediary.T:
                        time_left = self.intermediary.T - time() + start
#                       Sleep for long waiting time, active waiting is time left < 5ms
                        if time_left > 0.005:
                            # Small margin as OS operations may be time consuming
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
#           Single operation
            else:
                self.forward()

    def forward(self):
        """Controls plotting results and operation of buffering"""

        self.i += 1
#       Bar plots might be slow
        if self.i > Plotter.i + 3:
            Intermediary.instance.bar_toggle(False)
#       If problem occurs any way
        if self.i > Plotter.i + 5:
            Intermediary.instance.prolong(multiply=1.5)
            Plotter.i = self.i
#       Get new results
        try:
            self.intermediary.transform.analyse()
            Plotter.data = self.intermediary.transform.getHistoryA()
        except:
            return
#       Set data for plotter
        if Plotter.data.dtype == np.complex:
            Plotter.data = np.real(Plotter.data)
#       Signal
        self.w.emit(SIGNAL("plot()"))
#       Keeping track of progress
        if self.intermediary.transform.finished:
            self.intermediary.play = False
            showinfo("", "End of file")
        self.strt = time()


class Worker:
    """QT plotting window starter"""

    def __init__(self, ui, condition):

        self.app = QApplication([])
        self.plotter = Plotter(ui, condition)

        self.run()

    def run(self):

        sys.exit(self.app.exec_())


class Wrapper(QObject):
    """Enabling usage of QT signals"""

    signal = pyqtSignal(np.ndarray)

    def __init__(self):

        QObject.__init__(self)
        self.connect(self, SIGNAL("plot()"), Plotter.plot)


class Singer(threading.Thread):
    """
    Class handling audio playback. After creation awaits signal from condition variable
    """

    def __init__(self, cv, **kwarg):
        """:arg cv Condition variable signaling playback start"""

        super().__init__(**kwarg, target=self._sing, name="Music")
        self.condition = cv
        self.play_obj = None
        self.termination = False

    def _sing(self):
        """Plays signal read to transform class"""

#       Wait for condition variable signal
        while True:
            self.condition.acquire()
            self.condition.wait()
            self.condition.release()
            if self.termination:
                return
            ii = Intermediary.instance
            if ii.const:
                # New playback
                if self.play_obj is None:
                    self.play_obj = ii.transform.play(ii.volume, False, True, ii.transform.position)

    def terminate(self):
        """Schedule thread termination"""

        self.stop()
        self.termination = True
        self.condition.acquire()
        self.condition.notify()
        self.condition.release()

    def stop(self):
        """Stop current playback"""

#       Check if music was started
        if self.play_obj is not None:
            # Stop and await next signal
            self.play_obj.stop()
            self.play_obj = None
