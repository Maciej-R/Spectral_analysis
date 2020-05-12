import pyqtgraph as qg
from time import time
from PyQt5.QtCore import QThreadPool
from threading import Thread, Condition
from Intermediary import Intermediary, Worker, Plotter
import UI_tk


class UI:

    def __init__(self, pth):

        condition = Condition()
        self.tk = Thread(target=UI_tk.UI, args=[pth, condition])
        self.tk.start()
        condition.acquire()
        condition.wait()
        condition.release()
        self.intermediary = Intermediary.instance
        #self.WidgetPlot.setLogMode(True, False)
        self.worker = Worker(self, Intermediary.instance.cv_play)
        self.worker.run()


    # def run(self):
    #
    #
    #     self.ui_thread.start()
    #     self.worker.run()
        #self.worker.app.exec_()


if __name__ == '__main__':
    path = "/home/maciek/PycharmProjects/Spectral_analysis/UI_v1.ui"
    app = UI(path)

    # self.img = tk.Label(self.imageFrame, image=self.render)
    # self.img.image = render
    # self.img.place(x=0, y=0)
    # self.img.place()
    # self.end = self.start