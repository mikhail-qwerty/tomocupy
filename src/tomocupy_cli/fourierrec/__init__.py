from pkg_resources import get_distribution, DistributionNotFound

from tomocupy_cli.fourierrec.fourierrec import FourierRec
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass