import OT
import Fourier
import tkinter as tk
import re
import numpy as np


class Intermediary:
    """Class provides UI with data from backend and forwards user input to backend"""

    def __init__(self, ui):

        self.transform = None
        self.ui = ui
        self.builder = ui.builder
        self.Tlist = ["DCT-I", "DCT-II", "DCT-III", "Hadamard", "DFT", "DtFT", "FFT"]
        self.Clist = [OT.OT, OT.OT, OT.OT, OT.OT, Fourier.DFT, Fourier.DtFT, Fourier.FFT]
        self.Tdict = {(k, v) for k, v in zip(self.Tlist, self.Clist)}
        self.type = -1
        self.T = None
        self.play = False
        self.logx = True
        self.sound = False
        self.scale = True
        self.const = True
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

    def dispatch(self):

        if self.play:
            self.transform.analyse()
            #self.ui.plot()
            #self.transform.play()

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
            SN = self.builder.get_object("SN")
            SN.delete(0, len(SN.get()))
            SN.insert(0, str(self.transform.N))

    def start(self, event):
        """"""

        if self.transform is None:
            return
        self.play = True
        #self.transform.analyse()
        #self.ui.plot(None)
        #self.transform.play()

    def pause(self, event):
        """"""

        self.play = False

    def log_toggle(self, event):
        """"""

        if self.builder.get_variable("VarLog").get() == 0:
            self.logx = True
        else:
            self.logx = False

    def scale_toggle(self, event):
        """"""

        if self.builder.get_variable("VarScale").get() == 0:
            self.scale = True
        else:
            self.scale = False

    def const_toggle(self, event):
        """"""

        if self.builder.get_variable("VarConst").get() == 0:
            self.const = True
        else:
            self.const = False
