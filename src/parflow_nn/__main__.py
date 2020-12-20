"""
The parflow-nn package can be run as a module by invoking it as:
python -m parflow_nn <command> <arguments> ..

The commands currently supported are:

nn - Run the ConvLSTM model on a parflow run directory

"""

import sys
from parflow_nn.nn import main as nn


if __name__ == '__main__':

    commands = ('nn',)

    if len(sys.argv) < 2:
        print('Usage: python -m hatchet <command> <arguments ..>')
        print('The following commands are supported: ' + ' '.join(commands))
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]
    if command not in commands:
        print('The following commands are supported: ' + ' '.join(commands))
        sys.exit(1)

    globals()[command](args)
