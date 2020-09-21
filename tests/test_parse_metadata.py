import os.path
import pytest
from parflow_nn.pfmetadata import PFMetadata

this_dir = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(this_dir, 'data')


@pytest.fixture(scope="module")
def m():
    # A PFMetadata object available to all functions
    return PFMetadata(os.path.join(DATA_FOLDER, 'sample.out.pfmetadata'))


def test_ndim(m):
    # Attributes can be accessed by using the PFMetadata object as a dictionary
    assert m['ComputationalGrid.NX'] == m['ComputationalGrid.NX'] == 41
    assert m['ComputationalGrid.NZ'] == 50
