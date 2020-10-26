# Sources

- `country-by-population.json` - https://github.com/samayo/country-json/blob/master/src/country-by-population.json

# GIS data

<https://shop.swisstopo.admin.ch/en/products/landscape/boundaries3D>

    DATEN/swissBOUNDARIES3D/SHAPEFILE_LV03_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp

Converted to 2D WGS84 coordinates using QGIS (required by Basemap.readshapefile)

* Layer -> Add Layer -> Add Vector Layer
* Layer -> Save As...
  - CRS: Default CRS ... WGS 84
  - Geometry type: Polygons
  - Include z-dimension: uncheck

