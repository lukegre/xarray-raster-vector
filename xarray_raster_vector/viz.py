# example of useage: gdf.explore(**shp.GOOGLE_TERRAIN)
_LEAFLET_DEFAULTS = dict()
GOOGLE_TERRAIN = dict(tiles='http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}', attr='Google', **_LEAFLET_DEFAULTS)
GOOGLE_SATELLITE = dict(tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}', attr='Google', **_LEAFLET_DEFAULTS)