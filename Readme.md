ParFlow - NN
=======

Preliminary steps to train a ConvLSTM model of ParFlow

Installation
--------------------

`Parflow-NN` works on Python 3. An Anaconda environment with Python 3.8
is recommended until this code is available as a package.

Use the supplied `environment.yml` to create the environment.
```
conda env create -f environment.yml
conda activate parflow_nn
```

If you prefer to use `pip` in a `venv`, a `requirements.txt` is provided too, but keep
in mind that installing `tensorflow` through `conda` is likely to pose fewer compatibility
issues with the CUDA Toolkit.
```
pip install -r requirements.txt
```

After installing all dependencies, install `pfio` in the currently active environment.
```
pip install git+https://github.com/hoangtv1899/pfio.git
```

Simulation
--------------------

```
cd washita/tcl_scripts
tclsh LW_test.tcl
```

Converting ParFlow output to .nc files
--------------------
Please follow the python notebook file: ```preprocess/write_nc.ipynb```

Training ConvLSTM model
--------------------
Please follow the python notebook file: ```new_NN_arch.ipynb```

Contacts
--------------------
+ Hoang Tran (hoangtran@mines.edu; hoangtran@princeton.edu)