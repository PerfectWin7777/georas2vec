"""GeoRas2Vec — scanned geological map (raster) -> GIS vector map (GeoPackage): units, faults and names."""
import os as _os, sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)          # the modules import each other by flat name (pipeline, naming_model, ...)

__version__ = "0.6.0"

from .assets import fetch_assets  # noqa: E402,F401
