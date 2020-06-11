from threading import Thread, Condition
from Intermediary import Intermediary, Worker, Plotter
import UI_tk


class UI:
    """Runs tk GUI and Qt graph window"""

    def __init__(self, pth):

        # Tk readiness indication
        condition = Condition()
        self.tk = Thread(target=UI_tk.UI, args=[pth, condition])
        self.tk.start()
        condition.acquire()
        condition.wait()
        condition.release()
        self.intermediary = Intermediary.instance
#       Starting Qt plotter
        self.worker = Worker(self, Intermediary.instance.cv_play)
        self.worker.run()


if __name__ == '__main__':
    path = "UI_v1_mod.ui"
    app = UI(path)

