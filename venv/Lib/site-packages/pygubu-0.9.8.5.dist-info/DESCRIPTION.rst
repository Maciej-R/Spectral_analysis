
Welcome to pygubu a GUI designer for tkinter
============================================

Pygubu is a RAD tool to enable quick & easy development of user interfaces
for the python tkinter module.

The user interfaces designed are saved as XML, and by using the pygubu builder
these can be loaded by applications dynamically as needed.
Pygubu is inspired by Glade.


Installation
------------

Pygubu requires python >= 2.7 (Tested only in python 2.7.3 and 3.2.3
with tk8.5)

Download and extract the tarball. Open a console in the extraction
path and execute:

::

    python setup.py install


Usage
-----

Create an UI definition using pygubu and save it to a file. Then, 
create your aplication script as shown below. Note that 'mainwindow' 
is the name of your Toplevel widget.

::

    # helloworld.py
    import tkinter as tk
    import pygubu


    class HelloWorldApp:

        def __init__(self):

            #1: Create a builder
            self.builder = builder = pygubu.Builder()

            #2: Load an ui file
            builder.add_from_file('helloworld.ui')

            #3: Create the mainwindow
            self.mainwindow = builder.get_object('mainwindow')

        def run(self):
            self.mainwindow.mainloop()


    if __name__ == '__main__':
        app = HelloWorldApp()
        app.run()


See the examples directory or watch this hello world example on
video http://youtu.be/wuzV9P8geDg


