import rioxarray as rxr

from .conversion import (
    polygon_to_raster_bool, 
    polygons_to_raster_int,
    raster_bool_to_vector,
    raster_int_to_vector)

from . import viz
from . import vector
from . import raster
from . import projection
from . import utils
from . import accessors

from loguru import logger

accessors = (
    accessors.get_accessor_funcs(accessors.VectorRaster, 'df.rv') + ["=" * 23] +
    accessors.get_accessor_funcs(accessors.RasterVector, 'da.rv') + ["=" * 23] +
    accessors.get_accessor_funcs(accessors.ScikitImage, 'da.skimg'))

logger.debug("\nThe following accessors have been added:\n  " + '\n  '.join(accessors))