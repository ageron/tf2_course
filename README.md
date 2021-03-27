Deep Learning with TensorFlow 2 and Keras – Notebooks
==============================================

This project accompanies my **Deep Learning with TensorFlow 2 and Keras** trainings. It contains the exercises and their solutions, in the form of [Jupyter](http://jupyter.org/) notebooks.

If you are looking for the code accompanying my O'Reilly book, [Hands-on Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), visit this GitHub project: [ageron/handson-ml2](https://github.com/ageron/handson-ml2).

## Quick Start

### Want to play with these notebooks online without having to install anything?
Use any of the following services.

**WARNING**: Please be aware that these services provide temporary environments: anything you do will be deleted after a while, so make sure you download any data you care about.

* **Recommended**: open this repository in [Colaboratory](https://colab.research.google.com/github/ageron/tf2_course/blob/master/):
<a href="https://colab.research.google.com/github/ageron/tf2_course/blob/master/"><img src="https://colab.research.google.com/img/colab_favicon.ico" width="90" /></a>

* Or open it in [Binder](https://mybinder.org/v2/gh/ageron/tf2_course/master):
<a href="https://mybinder.org/v2/gh/ageron/tf2_course/master"><img src="https://matthiasbussonnier.com/posts/img/binder_logo_128x128.png" width="90" /></a>

  * _Note_: Most of the time, Binder starts up quickly and works great, but when tf2\_course is updated, Binder creates a new environment from scratch, and this can take quite some time.

* Or open it in [Deepnote](https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/tf2_course/blob/master/01_neural_nets_with_keras.ipynb):
<a href="https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/tf2_course/blob/master/01_neural_nets_with_keras.ipynb"><img src="https://www.deepnote.com/static/illustration.png" width="150" /></a>

### Just want to quickly look at some notebooks, without executing any code?

Browse this repository using [jupyter.org's notebook viewer](https://nbviewer.jupyter.org/github/ageron/tf2_course/blob/master/01_neural_nets_with_keras.ipynb):
<a href="https://nbviewer.jupyter.org/github/ageron/tf2_course/blob/master/01_neural_nets_with_keras.ipynb"><img src="https://jupyter.org/assets/nav_logo.svg" width="150" /></a>

_Note_: [github.com's notebook viewer](01_neural_nets_with_keras.ipynb) also works but it is slower and the math equations are not always displayed correctly.

### Want to install this project on your own machine?

Start by installing [Anaconda](https://www.anaconda.com/distribution/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), [git](https://git-scm.com/downloads), and if you have a TensorFlow-compatible GPU, install the [GPU driver](https://www.nvidia.com/Download/index.aspx), as well as the appropriate version of CUDA and cuDNN (see TensorFlow's documentation for more details).

Next, clone this project by opening a terminal and typing the following commands (do not type the first `$` signs on each line, they just indicate that these are terminal commands):

    $ git clone https://github.com/ageron/tf2_course.git
    $ cd tf2_course

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate tf2c
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

If you need further instructions, read the [detailed installation instructions](INSTALL.md).

# FAQ

**Which Python version should I use?**

I recommend Python 3.7. If you follow the installation instructions above, that's the version you will get. Most code will work with other versions of Python 3, but some libraries do not support Python 3.8 or 3.9 yet, which is why I recommend Python 3.7.

**I'm getting an SSL error on MacOSX**

You probably need to install the SSL certificates (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)). If you downloaded Python from the official website, then run `/Applications/Python\ 3.7/Install\ Certificates.command` in a terminal (change `3.7` to whatever version you installed). If you installed Python using MacPorts, run `sudo port install curl-ca-bundle` in a terminal.

**I've installed this project locally. How do I update it to the latest version?**

See [INSTALL.md](INSTALL.md)

**How do I update my Python libraries to the latest versions, when using Anaconda?**

See [INSTALL.md](INSTALL.md)

That's it! Now, have fun learning TensorFlow 2!
