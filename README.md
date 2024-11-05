# xarray-raster-vector
A lightweight set of tools for converting raster to vector and vice-versa.  
Uses xarray, geopandas and scikit-image as the main building blocks.

## Installation

```bash
pip install git+https://github.com/lukegre/xarray-raster-vector.git
```

## Usage

A very basic example is below, but see `demo.ipynb` for more comprehensive examples.
```python
import xarray_raster_vector as xrv

fname = 'somepath_to_a_mask.nc'
da_mask = xr.open_dataarray(fname).astype(bool)

# apply skimage cleaning protocol
da_mask_clean = da.rv.clean_mask()

# convert to polygons
df = da_mask_clean.rv.to_vector()
# convert to raster (xarray)
da = df.rv.to_raster(target_da=da_mask)

```
