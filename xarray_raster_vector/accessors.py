from xarray import register_dataarray_accessor
# import pandas equivalent to register accessor for pandas
from pandas.api.extensions import register_dataframe_accessor

from loguru import logger
from functools import wraps
from skimage import morphology, measure

from .utils import enable_xarray_wrapper
from .conversion import (
    raster_int_to_vector,
    raster_bool_to_vector,
    polygon_to_raster_bool,
    polygons_to_raster_int)


@register_dataarray_accessor("rv")
class RasterVector:
    def __init__(self, da) -> None:
        from .raster import prep_raster
        self._da = prep_raster(da)

    def to_vector(self, **kwargs):
        """
        Converts a rasterized mask to a vectorized representation.

        Parameters
        ----------
        buffer_dist : float, optional [0]
            Polygons can be smoothed by expanding and then shrinking
            the extent of the polygons. Should be in the same unit as
            the raster coordinates. If lat/lon, then must be in degrees.
        simplify_dist : float, optional [0]
            Polygons can be simplified by removing vertices. Should be
            in the same unit as the raster coordinates. If lat/lon, then
            must be in degrees.
        combine_polygons : bool, optional [False]
            If True, then polygons belonging to the same mask are combined
            into a single polygon. If False, then each connected component
            is a separate polygon. Only applies to boolean masks. 
        names : list, optional [None]
            If the mask is integer type, then a list of names for each class
            can be provided. The default is None, in which case the classes
            are numbered according to their integer value.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the vectorized representation of the mask.
        """        
        da = self._da
        
        if da.dtype == bool:
            df = raster_bool_to_vector(da, **kwargs)
        elif da.dtype == int:
            df = raster_int_to_vector(da, **kwargs)
        else:
            raise TypeError("DataArray must be type [int|bool]")
        
        return df
    
    def to_geotiff(self, filname:str, **kwargs):
        """
        Saves the DataArray as a geotiff file.

        Parameters
        ----------
        filname : str
            Output file name.
        kwargs : dict
            Additional keyword arguments to pass to the `rio.to_raster` method.
        """
        from .raster import save_raster_3d_to_geotiff
        
        da = self._da

        save_raster_3d_to_geotiff(da, filname, **kwargs)
        
    def get_bbox_latlon(self, as_geopandas=False):
        """
        Returns the bounding box of the DataArray in lat/lon coordinates.

        Parameters
        ----------
        as_pandas : bool, optional [False]
            If True, returns a pandas Series with the bounding box coordinates.
            If False, returns a tuple with the bounding box coordinates.

        Returns
        -------
        tuple|pd.Series
            A tuple with the bounding box coordinates in lat/lon.
        """
        from .raster import get_bounds_latlon
        from .vector import bbox_to_geopandas
        
        da = self._da
        bbox = get_bounds_latlon(da)
        
        if as_geopandas:
            bbox = bbox_to_geopandas(bbox, crs=da.rio.crs)
        
        return bbox

    def get_utm_code(self):
        """
        Returns the UTM EPSG code for the DataArray.

        Returns
        -------
        int
            The EPSG code for the UTM zone of the DataArray.

        Raises
        ------
        ValueError
            If the DataArray spans more than 12 degrees of longitude.
        Warning
            If the DataArray spans more than 6 degrees of longitude.
        """
        from .projection import compute_utm_from_lat_lon
        da = self._da

        lon = da[da.dims[-1]].values
        lat = da[da.dims[-2]].values
        x0, x1 = lon.min(), lon.max()

        if (x1 - x0) > 6:
            logger.warning("DataArray spans more than 6 degrees of longitude. UTM zones typically span 6 degrees.")
        if (x1 - x0) > 12:
            raise ValueError("DataArray spans more than 12 degrees longitude. UTM zones typically span 6 degrees.")

        lat_center = (lat.min() + lat.max()) / 2
        lon_center = (lon.min() + lon.max()) / 2

        epsg = compute_utm_from_lat_lon(lat_center, lon_center)

        return epsg

    def clean_mask(self, holes=64, objects=64, opening=None, closing=None):
        """
        Cleans the mask by removing small objects and holes.

        Parameters
        ----------
        holes : int, optional [64]
            The minimum area of holes to remove.
        objects : int, optional [64]
            The minimum area of objects to remove.

        Returns
        -------
        xr.DataArray
            A cleaned mask with small objects removed.
        """
        da = self._da

        assert da.dtype == bool, "DataArray must be boolean type."

        da = (
            da
            .morph.remove_small_holes(area_threshold=holes)
            .morph.remove_small_objects(min_size=objects)
            .morph.binary_opening()
            .morph.binary_closing())
        return da


@register_dataarray_accessor("skimg")
class ScikitImage:
    """
    Accessor for skimage.morphology operations:
    - binary_opening
    - binary_closing
    - binary_erosion
    - binary_dilation
    - remove_small_holes
    - remove_small_objects
    """
    def __init__(self, da):
        self._da = da

    @wraps(morphology.binary_opening)
    def binary_opening(self, **kwargs):
        func = enable_xarray_wrapper(morphology.binary_opening)
        return func(self._da, **kwargs)
    
    @wraps(morphology.binary_closing)
    def binary_closing(self, **kwargs):
        func = enable_xarray_wrapper(morphology.binary_closing)
        return func(self._da, **kwargs)
    
    @wraps(morphology.binary_erosion)
    def binary_erosion(self, **kwargs):
        func = enable_xarray_wrapper(morphology.binary_erosion)
        return func(self._da, **kwargs)
    
    @wraps(morphology.binary_dilation)
    def binary_dilation(self, **kwargs):
        func = enable_xarray_wrapper(morphology.binary_dilation)
        return func(self._da, **kwargs)
    
    @wraps(morphology.remove_small_holes)
    def remove_small_holes(self, **kwargs):
        func = enable_xarray_wrapper(morphology.remove_small_holes)
        return func(self._da, **kwargs)
    
    @wraps(morphology.remove_small_objects)
    def remove_small_objects(self, **kwargs):
        func = enable_xarray_wrapper(morphology.remove_small_objects)
        return func(self._da, **kwargs)

    @wraps(measure.label)
    def label(self, **kwargs):
        da = self._da

        return_num = kwargs.pop('return_num', False)
        
        func = enable_xarray_wrapper(measure.label, **kwargs)
        out = func(da)
        
        ser = out.to_series()
        counts = ser.value_counts()
        out.attrs['n_labels'] = len(counts)

        if return_num:
            out = out.to_dataset(name='labels')
            out['counts'] = counts.to_xarray().rename(index='label')
        
        return out


@register_dataframe_accessor("rv")
class VectorRaster:
    def __init__(self, df):
        self._df = df

    def _check_df(self):
        import geopandas as gpd

        df = self._df

        assert isinstance(df, gpd.GeoDataFrame), "Must be a GeoDataFrame"
        assert "geometry" in df.columns, "GeoDataFrame must have a 'geometry' column"

    @wraps(polygons_to_raster_int)
    def to_raster(self, da_target, **kwargs):
        """
        Convert the GeoDataFrame to a raster mask.

        Parameters
        ----------
        da_target : xr.DataArray
            The target grid to match the mask to. It is assumed that the 
            spatial dimensions [y, x] are in positions [-2, -1]. 
        by_column : str, optional
            The column in the GeoDataFrame to group the polygons by. If None, then each
            row is converted to a separate integer value. The default is None.
        kwargs : dict, optional
            Additional keyword arguments to pass to joblib.Parallel. The default is {}.

        Returns
        -------
        xr.DataArray
            A DataArray with the raster mask with integer values.
        """
        from .raster import prep_raster
        import xarray as xr
        
        df = self._df
        self._check_df()

        assert isinstance(da_target, xr.DataArray), "Must be a DataArray"
        
        geom = df.geometry
        da = prep_raster(da_target)

        if len(geom) == 1:
            mask = polygon_to_raster_bool(geom.iloc[0], da)
        elif len(geom) < 20:
            mask = polygons_to_raster_int(df, da, **kwargs)
        elif len(geom) >= 20:
            raise ValueError("Too many polygons to convert to raster.")
        elif len(geom) == 0:
            raise ValueError("No polygons to convert to raster.")
        
        return mask
    
    def crop_to_da(self, da):
        from .vector import clip_geodata_to_grid
        from .raster import prep_raster

        df = self._df
        da = prep_raster(da)

        return clip_geodata_to_grid(df, da)
    

def get_accessor_funcs(class_object, accessor_name):
    """
    Print all accessors for a given class object.
    """
    import inspect

    txt = []
    for name, obj in inspect.getmembers(class_object):
        if hasattr(obj, "__doc__"):
            if name.startswith("__"):
                continue
            txt += f"{accessor_name}.{name}",
    return txt