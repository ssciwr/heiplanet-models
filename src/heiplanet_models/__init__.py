from . import Jmodel, computation_graph, utils

# Import the version from the package metadata or provide a fallback
# if that is not possible
try:
    from importlib.metadata import version

    __version__ = version("heiplanet_models")
except ImportError:
    __version__ = "unknown"

# Optional: Define what gets imported with "from heiplanet_models import *"
__all__ = ["Jmodel", "__version__", "computation_graph", "utils"]
