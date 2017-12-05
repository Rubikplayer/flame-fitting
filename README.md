# FLAME Face Model

This codebase demonstrates how to load and play with FLAME, a lightweight and expressive generic face model to be presented in:

Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero, Learning a model of facial shape and expression from 4D scans, ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 2017

With this code, you can:
 * Load and evaluate FLAME model
 * Fit FLAME model to 3D landmarks

To request for FLAME model and registrations, please see the [project page](http://flame.is.tue.mpg.de)

This repo is maintained by [Tianye Li](https://sites.google.com/site/tianyefocus/). The codes in `smpl_webuser` are directly from [SMPL Python code](http://smpl.is.tue.mpg.de/).

### Dependencies

This code uses Python 2.7 and need the following dependencies:

- [numpy & scipy](http://www.scipy.org/scipylib/download.html)
- [opencv](http://opencv.org/)
- [chumpy](https://github.com/mattloper/chumpy)

### Set-up

Clone the git project:
```
$ git clone https://github.com/Rubikplayer/flame-fitting.git
```

Set up virtual environment:
```
$ mkdir <your_home_dir>/.virtualenvs
$ virtualenv --system-site-packages <your_home_dir>/.virtualenvs/flame
```

Activate virtual environment:
```
$ cd flame-fitting
$ source <your_home_dir>/.virtualenvs/flame/bin/activate
```

Update the PYTHONPATH environment variable so that the system knows how to find the SMPL code. Add the following lines to your ~/.bash_profile file (create it if it doesn't exist; Linux users might have ~/.bashrc file instead), set the location to where you clone the project to.
```
FLAME_LOCATION=<flame_project_dir>
export PYTHONPATH=$PYTHONPATH:$FLAME_LOCATION
```

and run:
```
$ source ~/.bash_profile
```

To install numpy, scipy and chumpy:
```
$ pip install numpy
$ pip install scipy
$ pip install chumpy
```
To deactivate the virtual environment:
```
$ deactivate
```

### Demo

See `hello_world.py` and `facefit_lmk3d.py` for the demos.

### Citing

Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero. 2017. Learning a model of facial shape and expression from 4D scans. ACM Trans. Graph. 36, 6, Article 194 (November 2017), 17 pages. https://doi.org/10.1145/3130800.3130813

### License

Free for non-commercial and scientific research purposes. By using this code, you acknowledge that you have read the terms and conditions (http://flame.is.tue.mpg.de/data_license), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code. You further agree to cite the FLAME paper when reporting results with this model.
