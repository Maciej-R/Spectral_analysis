import OT
import Fourier
import tkinter as tk
import re
import numpy as np
import threading
from time import time
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import multiprocessing as mp
import threading
from tkinter import Tk

class Intermediary:
    """Class provides UI with data from backend and forwards user input to backend"""

    def __del__(self):

        for i in range(self.nbuffs):
            self.exchange[i]["intermediary"] = None
            self.buffers[i].join(200)
            if self.buffers[i].is_alive():
                self.buffers[i].terminate()
            print("end")

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
        self.sound = False
        self.const = True
        self.singer = None
        self.dispatcher = None
        #self.shms = [shared_memory.SharedMemory(size=sys.getsizeof(DataExchange), create=True)]
        self.managers = [mp.Manager() for i in range(self.nbuffs)]
        self.exchange = list()
        for mng in self.managers:
            self.exchange.append(mng.dict({"ready": False, "render": None, "data": None, "position": None,
                                           "condition": mng.Condition(), "logx": True, "scale": True,
                                           "freqs": None}))
#        self.exchange = [mng.dict() for mng in self.managers]
#        self.data = [[None] for i in range(self.nbuffs)]
#        self.positions = [0 for i in range(self.nbuffs)]
#        self.conditions = [threading.Condition() for i in range(self.nbuffs)]
        self.buffers = [mp.Process(target=Render, name="Buffering", args=((self.exchange[i]), ))
                        for i in range(self.nbuffs)]
        for buf in self.buffers:
            buf.start()
        self.cbuff = 0
        self.pstart = time()
        self.pstop = time()
        self.cv_play = threading.Condition()
        self.render = None
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
        self.exchange_param("freqs", self.transform.freqs)

    def dispatch(self):
        """If program runs in constant audio playing mode than function calls every period
        of refreshing forward function"""

        start = time()
        while self.play:
            while time() - start < self.T - 0.0005:
                pass
            try:
                self.forward()
            except RuntimeError as e:
                tk.messagebox.showerror("", e)
                self.play = False
                break
            #print(time() - prev)
            start = time()

    def forward(self):
        """Controls plotting results and operation of buffering"""

        self.exchange[self.cbuff]["condition"].acquire()
        if not self.exchange[self.cbuff]["ready"]:
            pass  #raise RuntimeError("Buffering too slow")
        self.ui.plot(self.exchange[self.cbuff]["render"])
        start = time()
        self.exchange[self.cbuff]["position"] = self.transform.N
        self.transform.analyse()
        self.exchange[self.cbuff]["data"] = self.transform.getHistoryA()
        self.exchange[self.cbuff]["ready"] = False
        self.exchange[self.cbuff]["condition"].notify()
        self.exchange[self.cbuff]["condition"].release()
        #print(time()-start)
        self.cbuff += 1
        self.cbuff %= self.nbuffs

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
        self.exchange_param("freqs", self.transform.freqs)
        if None is self.exchange[0]["data"]:
            self.init_buffers()
        else:
            self.purge_buffers()


    def start(self, event):
        """Start dispatching thread"""

        if self.transform is None or self.play:
            return
        self.play = True
        self.dispatcher = threading.Thread(target=self.dispatch, name="Dispatcher")
        self.dispatcher.start()

    def pause(self, event):
        """Stops dispatching thread"""

        self.play = False

    def log_toggle(self, event):
        """Change use of logarithmic scale indicator"""

        if self.builder.get_variable("VarLog").get() == 0:
            self.exchange_param("logx", True)
        else:
            self.exchange_param("logx", False)

    def scale_toggle(self, event):
        """Change use of scaling indicator"""

        if self.builder.get_variable("VarScale").get() == 0:
            self.exchange_param("scale", True)
        else:
            self.exchange_param("scale", False)

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
        self.forward()

    def sing(self):
        """Plays N music samples starting from current position in signal processing"""

        #self.transform.play(self.volume, False, True)
        while self.sound:
            self.cv_play.acquire()
            self.cv_play.wait()
            self.cv_play.release()
            self.transform.play(self.volume, False)

    def init_buffers(self):
        """Fill buffers at beginning"""

        for i in range(self.nbuffs):

            self.exchange[i]["position"] = self.transform.N
            self.transform.analyse()
            self.exchange[i]["condition"].acquire()
            self.exchange[i]["data"] = self.transform.getHistoryA()
            self.exchange[i]["condition"].notify()
            self.exchange[i]["condition"].release()

    def purge_buffers(self):
        """If buffering threads were already running call this instead of init_buffers.
        This function fills buffers with data from new input"""

        for ex in self.exchange:

            ex["condition"].acquire()
            ex.render = None
            ex.ready = False
            ex.position = self.transform.N
            self.transform.analyse()
            ex.data = self.transform.getHistoryA()
            ex["condition"].notify()
            ex["condition"].release()

    def exchange_param(self, param, value):

        for ex in self.exchange:

            ex[param] = value


class Render:
    """Prepares and stores data and render"""

    def __init__(self, exchange):
        self.fig = None
        self.ax = None
        self.running = True
        self.im = None
        self.prev_log = False
        self.exchange = exchange
        self.condition = self.exchange["condition"]
        self.data = self.exchange["data"]

        self.fig = Figure()
        #self.ax = self.fig.add_subplot(1, 1, 1)
        #self.ax.set_xlabel("Frequencies")
        self.fig.suptitle("Transform")

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()

        self.run()

    def render_fig(self):
        """
        Convert a Matplotlib figure to a 4D numpy array with RGBA channels
        @return Numpy 4D array of RGBA values
        """

        #start = time()
        self.plot(self.exchange["data"])
#       Draw the renderer
        #print(time() - start)
        #start = time()
        self.canvas.draw()
        #print(time() - start)
        #Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.array(self.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        #buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        #print(time() - start)
        w, h, d = buf.shape
        img = Image.frombytes("RGBA", (w, h), buf.tostring())
        self.exchange["render"] = img
        #print(time() - start)
        self.fig.delaxes(self.ax)
        #print(time() - start)
        self.exchange["ready"] = True
        #print(time()-start)

    def plot(self, plotdata):
        """"""

        self.ax = self.fig.add_subplot(111)
        #start = time()
        M = max(abs(plotdata))
        if self.exchange["scale"]:
            self.ax.set_ylim(-1, 1)
            if M < 1:
                M = 1
            plotdata /= M
        else:
            self.ax.set_ylim(-M, M)
        #print(time() - start)
        if self.exchange["logx"]:
            #if self.prev_log and self.im is not None:
                #self.im.set_ydata(plotdata)
                #self.ax.draw_aritst(self.ax.patch)
                #self.ax.draw_artist(self.im)
                #self.fig.canvas.update()
                #self.fig.canvas.flush_events()
            #else:
            self.im = self.ax.semilogx(self.exchange["freqs"], plotdata)
                #self.canvas.draw()
        else:
            self.im = self.ax.plot(self.exchange["freqs"], plotdata)
        #self.canvas.draw_idle()
        #print(time() - start)

    def run(self):

        while self.running is not None:
            self.condition.acquire()
            self.condition.wait()
            self.render_fig()
            #print(time() - start)
            self.condition.release()

    def stop(self):

        self.running = False


class DataExchange:

    def __init__(self, transform):
        self.logx = True
        self.scale = True
        self.transform = transform

