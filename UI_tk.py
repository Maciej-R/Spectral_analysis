import tkinter as tk
import pygubu
import Intermediary


class UI:
    """GUI"""

    def __init__(self, pth, cnd):
        """Initialize GUI"""

        # pygubu builder
        self.builder = pygubu.Builder()
        # Load an ui file
        self.builder.add_from_file(pth)
        self.intermediary = Intermediary.Intermediary(self)
        # Create the mainwindow
        self.mainFrame = self.builder.get_object('FrameMain')

        self.init_ui()
#        Notify about finished initialization
        cnd.acquire()
        cnd.notify()
        cnd.release()
#       Run GUI
        self.mainFrame.mainloop()

    def init_ui(self):
        """Setting parameters, adding listeners"""

        slider = self.builder.get_object("Volume")
        slider.to = 1
        slider.from_ = 0
        slider.set(0.2)
        self.list_transforms()
        self.add_listeners()

    def list_transforms(self):
        """Listing available transform in listbox"""

        Ltransforms = self.builder.get_object("TransformType")
        for entry in self.intermediary.Tlist:
            Ltransforms.insert(tk.END, entry)

    def add_listeners(self):
        """Connect event listeners to GUI elements"""

        confirm = self.builder.get_object("BConfirm")
        confirm.bind("<Button-1>", self.intermediary.type_confirmed)

        file = self.builder.get_object("Path")
        file.bind("<<PathChooserPathChanged>>", self.intermediary.file_selected)

        start = self.builder.get_object("BStart")
        start.bind("<Button-1>", self.intermediary.start)

        pause = self.builder.get_object("BPause")
        pause.bind("<Button-1>", self.intermediary.pause)

        logx = self.builder.get_object("CBLogX")
        logx.bind("<Button-1>", self.intermediary.log_toggleX)
        logx.select()

        logy = self.builder.get_object("CBLogY")
        logy.bind("<Button-1>", self.intermediary.log_toggleY)
        logy.select()

        scale = self.builder.get_object("CBScale")
        scale.bind("<Button-1>", self.intermediary.scale_toggle)
        scale.deselect()

        const = self.builder.get_object("CBConst")
        const.bind("<Button-1>", self.intermediary.const_toggle)
        const.select()

        next = self.builder.get_object("BNext")
        next.bind("<Button-1>", self.intermediary.next)

        sound = self.builder.get_object("CBSound")
        sound.bind("<Button-1>", self.intermediary.sound_toggle)
        sound.deselect()

        use_window = self.builder.get_object("CBUse")
        use_window.bind("<Button-1>", self.intermediary.window_toggle)
        use_window.select()

        trim = self.builder.get_object("CBTrim")
        trim.bind("<Button-1>", self.intermediary.trim_toggle)

        reset = self.builder.get_object("BReset")
        reset.bind("<Button-1>", self.intermediary.reset)

        change = self.builder.get_object("BChange")
        change.bind("<Button-1>", self.intermediary.attenuation_change)

        bar = self.builder.get_object("CBBar")
        bar.bind("<Button-1>", self.intermediary.bar_toggle)
