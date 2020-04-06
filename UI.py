import tkinter as tk
import pygubu
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.pyplot import subplots
import tkinter
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from matplotlib.animation import FuncAnimation
import Intermediary
from time import time
import numpy as np
import matplotlib.backends.backend_tkagg as tkagg


class UI:

    def __init__(self, pth):

        #pygubu builder
        self.builder = pygubu.Builder()
        #Load an ui file
        self.builder.add_from_file(pth)
        self.intermediary = Intermediary.Intermediary(self, 8)
        # 3: Create the mainwindow
        self.mainFrame = self.builder.get_object('FrameMain')
        self.imageFrame = self.builder.get_object("GraphsFrame")
        self.img = self.builder.get_object("LImage")
        self.photo = None

        self.init_ui()
        self.i = 0
        self.AT = self.builder.get_object("EAudioTime")

    def run(self):

        self.mainFrame.mainloop()

    def plot(self, img):
        """Function regularly called by matplotlib.animation"""

        if self.photo is None:
            self.photo = ImageTk.PhotoImage(img)
            self.img.configure(image=self.photo)
        else:
            self.photo.paste(img)
        self.i += 1
#       Clear
        self.AT.delete(0, len(self.AT.get()))
#       Set new value
        self.AT.insert(0, str(self.i * self.intermediary.T))

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

        log = self.builder.get_object("CBLog")
        log.bind("<Button-1>", self.intermediary.log_toggle)
        log.select()

        scale = self.builder.get_object("CBScale")
        scale.bind("<Button-1>", self.intermediary.scale_toggle)
        scale.select()

        const = self.builder.get_object(("CBConst"))
        const.bind("<Button-1>", self.intermediary.const_toggle)
        const.select()

        next = self.builder.get_object("BNext")
        next.bind("<Button-1>", self.intermediary.next)

        sound = self.builder.get_object("CBSound")
        sound.bind("<Button-1>", self.intermediary.sound_toggle)
        sound.deselect()


if __name__ == '__main__':
    path = "UI_v1.ui"
    app = UI(path)
    app.run()

    # self.img = tk.Label(self.imageFrame, image=self.render)
    # self.img.image = render
    # self.img.place(x=0, y=0)
    # self.img.place()
    # self.end = self.start