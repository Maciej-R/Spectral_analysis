from cx_Freeze import setup, Executable
import os
import scipy
import pyqtgraph
import PySide2
import shiboken2

scipy_path = os.path.dirname(scipy.__file__)
qg = os.path.dirname(pyqtgraph.__file__)
ps = os.path.dirname(PySide2.__file__)
sb = os.path.dirname(shiboken2.__file__)
includefiles_list = [scipy_path, qg, ps, sb]

includes = []

setup(name="Rysnik_Projekt",
      version="1",
      description="",
      options={"build_exe": {"includes": includes, "include_files": includefiles_list}},
      executables=[Executable("UI.py")])