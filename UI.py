import tkinter as tk
import pygubu
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.pyplot import subplots
import tkinter
from matplotlib.animation import FuncAnimation
import Intermediary
from time import time
import numpy as np


class UI:

    def __init__(self, pth):

        #pygubu builder
        self.builder = pygubu.Builder()
        #Load an ui file
        self.builder.add_from_file(pth)
        self.intermediary = Intermediary.Intermediary(self)
        # 3: Create the mainwindow
        self.mainFrame = self.builder.get_object('FrameMain')
        self.imageFrame = self.builder.get_object("GraphsFrame")
        #Matplotlib in tk frame
        self.fig, self.ax = subplots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.imageFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        #self.toolbar = NavigationToolbar2Tk(self.canvas, self.imageFrame)
        #self.toolbar.update()
        #self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.animation = None
        self.T = 1
        self.start = time()
        self.end = time()

        #self.builder.connect_callbacks(self.intermediary)
        self.init_ui()
        self.animation = FuncAnimation(self.fig, self.plot, interval=25)

    def run(self):

        self.mainFrame.mainloop()

    def plot(self, frame):

        if not self.intermediary.play:
            return
        self.intermediary.transform.analyse()
        self.start = time()
        self.ax.clear()
        self.fig.suptitle("Transform")
        self.ax.set_xlabel("Frequencies")
        #self.ax.set_xlim(0, self.intermediary.transform.freqs[self.intermediary.transform.N-1])
        self.ax.set_ylim(-1, 1)
        his = self.intermediary.transform.getHistoryA()
        self.ax.semilogx(self.intermediary.transform.freqs, his/max(abs(his)))
        #self.canvas.draw()
        self.end = time()
        print(self.end - self.start)

    def set_animation(self):

        self.T = round(self.intermediary.transform.N/self.intermediary.transform.fs)
        self.animation = FuncAnimation(self.fig, self.plot,  interval=self.T)

    def stop_animaation(self):

        self.animation = None

    def init_ui(self):

        slider = self.builder.get_object("Volume")
        slider.to = 1
        slider.from_ = 0
        slider.set(0.2)
        self.list_transforms()
        self.add_listeners()

    def list_transforms(self):

        Ltransforms = self.builder.get_object("TransformType")
        for entry in self.intermediary.Tlist:
            Ltransforms.insert(tk.END, entry)

    def add_listeners(self):

        confirm = self.builder.get_object("BConfirm")
        confirm.bind("<Button-1>", self.intermediary.type_confirmed)

        file = self.builder.get_object("Path")
        file.bind("<<PathChooserPathChanged>>", self.intermediary.file_selected)

        start = self.builder.get_object("BStart")
        start.bind("<Button-1>", self.intermediary.start)

        pause = self.builder.get_object("BPause")
        pause.bind("<Button-1>", self.intermediary.pause)



if __name__ == '__main__':
    path = "UI_v1.ui"
    app = UI(path)
    app.run()