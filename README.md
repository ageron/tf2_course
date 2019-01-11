Deep Learning with TensorFlow 2 and Keras – Notebooks
==============================================

This project accompanies my **Deep Learning with TensorFlow 2 and Keras** trainings. It contains the exercises and their solutions, in the form of [Jupyter](http://jupyter.org/) notebooks.

**WARNING**: TensorFlow 2.0 preview may contain bugs and may not behave exactly like the final 2.0 release. Hopefully this code will run fine once TF 2 is out. This is extreme bleeding edge stuff people! :)

During the course itself, a URL will be provided for running the notebooks. You can participate in the course without installing anything local. If you prefer to work on a local installation, please follow the installation instructions below.

If you are looking for the code accompanying my O'Reilly book, [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do), visit this GitHub project: [handson-ml](https://github.com/ageron/handson-ml).

# Installation

First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/tf2_course.git
    $ cd tf2_course

If you are familiar with Python and you know how to install Python libraries, go ahead and install NumPy, Matplotlib, Jupyter and TensorFlow (see `requirements.txt` for details), and jump to the [Starting Jupyter](#starting-jupyter) section. If you need detailed instructions, read on.

You obviously need Python. Python 2 is already preinstalled on most systems nowadays, and sometimes even Python 3. You can check which version(s) you have by typing the following commands:

    $ python --version   # for Python 2
    $ python3 --version  # for Python 3

This course requires Python 3.5 or Python 3.6. TensorFlow does not support Python 3.7 yet. You may be able to run this code on Python 2, with minor tweaks, but it is deprecated so you really should upgrade to Python 3 now. To install Python 3.6, you have several options: on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3

Another option is to download and install [Anaconda](https://www.continuum.io/downloads). This is a package that includes both Python and many scientific libraries. You should prefer the Python 3.5 or 3.6 version.

## Using Anaconda

**Warning**: TensorFlow 2.0 preview is not available yet on Anaconda.

If you chose to install Anaconda, you can optionally create an isolated Python environment dedicated to this course. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this course), with potentially different libraries and library versions:

    $ conda create -n tf2course python=3.6 anaconda
    $ source activate tf2course

This creates a fresh Python 3.6 environment called `tf2course`, and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib and Jupyter), except for TensorFlow, so let's install it:

    $ conda install -n tf2course -c conda-forge tensorflow=2.0.0

This installs TensorFlow 2.0.0 in the `tf2course` environment (fetching it from the `conda-forge` repository). If you chose not to create a `tf2course` environment, then just remove the `-n tf2course` option.

You are all set! Next, jump to the [Starting Jupyter](#starting-jupyter) section.

## Using pip 
If you are not using Anaconda, you need to install several scientific Python libraries that are necessary for this course: NumPy, Jupyter, Matplotlib and TensorFlow. For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between the Python libraries versions and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries.

First you need to make sure you have the latest version of pip installed:

    $ pip3 install --user --upgrade pip

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo pip3` instead of `pip3` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. As explained above, this is recommended as it makes it possible to have a different environment for each project (e.g. one for this course), with potentially very different libraries, and different versions:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment using Python 3. If you have multiple versions of Python 3 installed on your system, you can replace \``which python3`\` with the path to the Python executable you prefer to use.

Now you want to activate this environment. You will need to run this command every time you want to use it.

    $ source ./env/bin/activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (or else you will probably need administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ pip3 install --upgrade -r requirements.txt

Great! You're all set, you just need to start Jupyter now.

## Starting Jupyter
To start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree).

Next, just click on any `*.ipynb` to open a Jupyter notebook.

That's it! Now, have fun learning TensorFlow 2!
