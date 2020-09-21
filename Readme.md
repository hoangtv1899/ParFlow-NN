![CI](https://github.com/hoangtv1899/ParFlow-NN/workflows/CI/badge.svg)

ParFlow - NN
=======

Preliminary steps to train a ConvLSTM model of ParFlow

## Installation

`Parflow-NN` works on Python 3. Creating an Anaconda environment with Python 3.8
is the easiest option, **especially if you want GPU support**.

### Anaconda

Install [anaconda](https://www.anaconda.com/products/individual) or
[miniconda](https://docs.conda.io/en/latest/miniconda.html) for your platform
and then use the supplied `environment.yml` to create and activate the environment.
Install the parflow-nn package inside this environment.
```
conda env create -f environment.yml
conda activate parflow_nn
pip install -e .
```

We recommend installing the `parflow-nn` package in `develop` mode (the `-e` flag in `pip install -e .`)
while this package is actively under development, so that you can tweak your local copy of the code
easily if you need to, and observe the changes.

If you wish to leverage an available GPU, install the `tensorflow-gpu` package.
```
conda install tensorflow-gpu
```

### Virtualenv

If you prefer to use `pip` in a `venv`, create the virtual environment, activate it, and then install the package.
```
python3 -m venv env
source env/bin/activate
pip install -e .
```

We recommend installing the `parflow-nn` package in `develop` mode (the `-e` flag in `pip install -e .`)
while this package is actively under development, so that you can tweak your local copy of the code
easily if you need to, and observe the changes.

If you wish to leverage an available GPU, first install and configure the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your platform. You're most
likely to succeed if you use the [10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
version of the toolkit, as this is the version that Tensorflow on PyPI seems to work best with.

Then install the `tensorflow-gpu` package through pip.
```
pip install tensorflow-gpu
```

#### Is my GPU detected properly?

After activating your Conda environment or venv, run the command
```
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```
and look at the last line of the output. If you see a `True`, you're all set!

Simulation
--------------------

```
cd washita/tcl_scripts
./model.tcl <run_name>
```

This folder generates the folder `runs/<run_name>` with the simulation results.

Training ConvLSTM model
--------------------
Execute the command:
```
python -m parflow_nn nn <run_dir>
```

This step requires you to point to the *run_directory*, i.e. the `runs/<run_name>` folder generated in the simulation
step above. If not already present, an intermediate folder called `nc_files` is created inside the `runs/<run_name>`
folder with the model `.nc` files.

Contacts
--------------------
+ Hoang Tran (hoangtran@mines.edu; hoangtran@princeton.edu)