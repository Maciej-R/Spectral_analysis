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
        self.worker = Worker(self, Intermediary.instance.cv_play)
        self.worker.run()


if __name__ == '__main__':
    path = "/home/maciek/PycharmProjects/Spectral_analysis/UI_v1_mod.ui"
    app = UI(path)

