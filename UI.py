import tkinter as tk
import pygubu
import Intermediary
import pyqtgraph as qg
from time import time
from PyQt5.QtCore import QThreadPool


class UI:

    def __init__(self, pth):

        #pygubu builder
        self.builder = pygubu.Builder()
        #Load an ui file
        self.builder.add_from_file(pth)
        self.intermediary = Intermediary.Intermediary(self)
        #self.intermediary.dispatcher.start()
        # 3: Create the mainwindow
        self.mainFrame = self.builder.get_object('FrameMain')

        #self.WidgetPlot.setLogMode(True, False)
        self.pool = QThreadPool()
        self.worker = Intermediary.Worker(self, self.intermediary.cv_play)
        self.pool.start(self.worker)

        self.init_ui()

    def run(self):

        self.mainFrame.mainloop()

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

        const = self.builder.get_object("CBConst")
        const.bind("<Button-1>", self.intermediary.const_toggle)
        const.select()

        next = self.builder.get_object("BNext")
        next.bind("<Button-1>", self.intermediary.next)

        sound = self.builder.get_object("CBSound")
        sound.bind("<Button-1>", self.intermediary.sound_toggle)
        sound.deselect()


class Plotter:

    def __init__(self, ui):

        self.ui = ui
        self.intermediary = self.ui.intermediary

        self.WindowGraphs = qg.GraphicsWindow()
        self.WidgetPlot = self.WindowGraphs.addPlot()

        # self.i = 0
        # self.AT = self.ui.builder.get_object("EAudioTime")

    def plot(self, data):
        """Function regularly called by matplotlib.animation"""

        start = time()
        self.WidgetPlot.clear()
        self.WidgetPlot.plot(self.intermediary.transform.freqs, data)
        print(time() - start)

# #       Clear
#         self.AT.delete(0, len(self.AT.get()))
# #       Set new value
#         self.AT.insert(0, str(self.i * self.intermediary.T))
#         self.i += 1


if __name__ == '__main__':
    path = "UI_v1.ui"
    app = UI(path)
    app.run()

    # self.img = tk.Label(self.imageFrame, image=self.render)
    # self.img.image = render
    # self.img.place(x=0, y=0)
    # self.img.place()
    # self.end = self.start