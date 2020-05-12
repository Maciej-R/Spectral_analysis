import tkinter as tk
import pygubu
import Intermediary


class UI:

    def __init__(self, pth, cnd):

        #pygubu builder
        self.builder = pygubu.Builder()
        #Load an ui file
        self.builder.add_from_file(pth)
        self.intermediary = Intermediary.Intermediary(self)
        #self.intermediary.dispatcher.start()
        # 3: Create the mainwindow
        self.mainFrame = self.builder.get_object('FrameMain')

        self.init_ui()
        cnd.acquire()
        cnd.notify()
        cnd.release()
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

