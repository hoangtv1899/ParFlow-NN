"""
Configuration options are available across the package by importing the config object from the base package.
These values are derived from a config.ini file packaged with the source code.
"""
import os
from parflow_nn import config


def test_config_string():
    # In the general case, config gives us access to the <section>.<key> value as strings
    assert config.netcdf.format == 'NETCDF4'


def test_config_boolean():
    # A value of True/False can be used to indicate python boolean types
    assert config.nn.batch_norm in (True, False)


def test_config_override():
    # Environment variables can be used to override config values, without modifying the config.ini file
    # Env vars follow the format PARFLOWNN_<SECTION>_<VARIABLE>
    old_value = os.getenv('PARFLOWNN_NETCDF_FORMAT')
    os.environ['PARFLOWNN_NETCDF_FORMAT'] = 'some_overridden_value'
    assert config.netcdf.format == 'some_overridden_value'

    if old_value is not None:
        os.environ['PARFLOWNN_NETCDF_FORMAT'] = old_value

