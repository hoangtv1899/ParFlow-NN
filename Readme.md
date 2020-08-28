ParFlow - NN
=======

Preliminary steps to train a ConvLSTM model of ParFlow

Simulation
--------------------

```
cd washita/tcl_scripts
tclsh LW_test.tcl

```

Converting ParFlow output to .nc files
--------------------
Please follow the python notebook file: ```preprocess/write_nc.ipynb```
Dependencies:
+ pfio (https://github.com/hoangtv1899/pfio)
+ netCDF4
+ numpy

Trainning ConvLSTM model
--------------------
Please follow the python notebook file: ```new_NN_arch.ipynb```

Dependencies:
+ Tensorflow

Contacts
--------------------
+ Hoang Tran (hoangtran@mines.edu; hoangtran@princeton.edu)