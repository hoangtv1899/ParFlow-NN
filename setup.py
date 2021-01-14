from setuptools import setup

setup(
    name='parflow-nn',
    version='0.0.2',
    packages=['parflow_nn'],
    package_dir={'': 'src'},
    package_data={'parflow_nn': ['config.ini']},
    zip_safe=True,

    python_requires='>=3.7',

    install_requires=[
        'cond-rnn',
        'configargparse',
        'fire',
        'graphviz',
        'h5py',
        'ipykernel',
        'jupyter',
        'keras',
        'matplotlib',
        'netcdf4',
        'numpy<1.19.0',
        'parflowio',
        'pydot',
        'scipy==1.4.1',
        'seaborn',
        'tensorflow',
        'tqdm',
        'xarray'
    ],

    extras_require = {
        'dev': ['pytest', 'mock']
    }

)
