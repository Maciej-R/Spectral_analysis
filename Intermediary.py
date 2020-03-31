import OT
import Fourier
import tkinter as tk
import re
import numpy as np
import threading
from time import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


class Intermediary:
    """Class provides UI with data from backend and forwards user input to backend"""

    def __init__(self, ui, nbuffs):

        self.transform = None
        self.ui = ui
        self.nbuffs = nbuffs
        self.builder = ui.builder
        self.Tlist = ["DCT-I", "DCT-II", "DCT-III", "Hadamard", "DFT", "DtFT", "FFT"]
        self.Clist = [OT.OT, OT.OT, OT.OT, OT.OT, Fourier.DFT, Fourier.DtFT, Fourier.FFT]
        self.Tdict = {(k, v) for k, v in zip(self.Tlist, self.Clist)}
        self.type = -1
        self.volume = 0.2
        self.T = 1
        self.play = False
        self.logx = True
        self.sound = False
        self.scale = True
        self.const = True
        self.singer = None
        self.dispatcher = None
        self.data = [[None] for i in range(self.nbuffs)]
        self.conditions = [threading.Condition() for i in range(self.nbuffs)]
        self.buffers = [threading.Thread(target=Render, name="Buffering", args=[self, self.conditions[i],
                                                                                self.data[i]])
                        for i in range(self.nbuffs)]
        self.pstart = time()
        self.pstop = time()
        self.cv_play = threading.Condition()
#       Getting references to UI widgets
        #self.SN = self.builder.get_object("SN")
        #self.LBTransform = self.builder.get_object("TransformType")

    def init_transform(self, str):
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

    def dispatch(self):

        i = 0
        while self.play:
            while time() - self.pstart < self.T - 0.0005:
                pass
            self.transform.analyse()
            if i % 2 == 0:
                self.buffer1.render_fig(self.transform.getHistoryA())
                self.ui.plot(self.buffer2.render)
            else:
                self.buffer2.render_fig(self.transform.getHistoryA())
                self.ui.plot(self.buffer1.render)
            i += 1
            self.pstart = time()
            #self.cv_play.acquire()
            #self.cv_play.notify()
            #self.cv_play.release()
            print(self.pstop - self.pstart)
            self.pstop = self.pstart

    def file_selected(self, event):
        """"""
#       Check if transform class exists. If not prompt and return
        if self.transform is None:
            tk.messagebox.showinfo("", "Choose transform type first")
            return
#       Check extension and read audio file
        pth = self.builder.get_object("Path").cget("path")
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
#       Initialize buffers
        self.init_buffers()

    def start(self, event):
        """"""

        if self.transform is None or self.play:
            return
        self.play = True
        self.dispatcher = threading.Thread(target=self.dispatch, name="Dispatcher")
        self.dispatcher.start()
        #self.transform.analyse()
        #self.ui.plot(None)
        #self.transform.play()

    def pause(self, event):
        """"""

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
            self.scale = True
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
        self.ui.single_plot()

    def sing(self):
        """Plays N music samples starting from current position in signal processing"""

        #self.transform.play(self.volume, False, True)
        while self.sound:
            self.cv_play.acquire()
            self.cv_play.wait()
            self.cv_play.release()
            self.transform.play(self.volume, False)

    def init_buffers(self):

        for i in range(self.nbuffs):

            self.buffers[i].start()
            self.transform.analyse()
            self.conditions[i].acquire()
            self.data[i][0] = self.transform.getHistoryA()
            self.conditions[i].notify()
            self.conditions[i].release()


class Render:
    """Prepares and stores data and render"""

    def __init__(self, intermediary, cv, data):
        self.fig = None
        self.ax = None
        self.render = None
        self.running = True
        self.data = data
        self.cv = cv
        self.intermediary = intermediary

    def render_fig(self):
        """
        Convert a Matplotlib figure to a 4D numpy array with RGBA channels
        @return Numpy 4D array of RGBA values
        """
        start = time()
        self.plot(self.data[0])

        # draw the renderer
        self.fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)

        w, h, d = buf.shape
        img = Image.frombytes("RGBA", (w, h), buf.tostring())
        self.render = ImageTk.PhotoImage(img)
        plt.close(self.fig)
        print(time() - start)

    def plot(self, plotdata):
        """"""

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.suptitle("Transform")
        self.ax.set_xlabel("Frequencies")
        M = max(abs(plotdata))
        if self.intermediary.scale:
            self.ax.set_ylim(-1, 1)
            if M < 1:
                M = 1
            plotdata /= M
        else:
            self.ax.set_ylim(-M, M)
        if self.intermediary.logx:
            self.ax.semilogx(self.intermediary.transform.freqs, plotdata)
        else:
            self.ax.plot(self.intermediary.transform.freqs, plotdata)

    def run(self):

        while self.running:
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
            self.render()

    def stop(self):

        self.running = False
