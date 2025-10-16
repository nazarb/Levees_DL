# Dataset

## Google Earth Engine code
The code used in this stage is available using following [link](https://github.com/nazarb/2025_levees_DL/tree/main/Pre_processing)
## Dataset composition

The final version of the dataset consist of 48 bands, which are calculated using multispelctral [LANDSAT-5](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2?hl=pl), [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED?hl=pl), [Sentinel 1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD?hl=pl) SAR images, and [TanDEM-X 30 EDEM](https://geoservice.dlr.de/web/dataguide/tdm30/).

The code for creating the composite consist of several parts
- mean value of each band of LANDSAT 5 calculated in dry season [[1]](#1)[[2]](#2)
- SMTVI - Seasonal Multi-Temporal Vegetation Indices (SMTVI) using EVI - LANDSAT 5 [[1]](#1)[[2]](#2)[[3]](#3)
- PCA of SMTVI - LANDSAT 5  (this study)
- Standard deviation of SMTVI - LANDSAT 5 (this study)
- NDMI - Normalized Difference Moisture Index (NDMI)- LANDSAT 5 [[4]](#4)
- PCA of NDMI - LANDSAT 5  (this study)
- Standard deviation of NDMI - LANDSAT 5 (this study)
- NDWI - Normalized Difference Water Index (NDWI) - LANDSAT 5  [[5]](#5)[[6]](#6)
- PCA of NDWI - LANDSAT 5  (this study)
- Standard deviation of NDWI - LANDSAT 5 (this study)
- mean value of VV band od Sentinel-1 calculated in dry season [[2]](#2)
- PCA of SMT calculated from VV band od Sentinel-1 (this study)
- Standard deviation of SMT calculated from VV band od Sentinel-1 (this study)
- mean value of VH band od Sentinel-1 calculated in dry season [[2]](#2)
- PCA of SMT calculated from VH band od Sentinel-1 (this study)
- Standard deviation of SMT calculated from VH band od Sentinel-1 (this study)
- mean value of each band of Sentinel-2 calculated in dry season [[2]](#2)[[3]](#3)
- mean value of Misra Soil Brightness Index (MSBI) calculated in dry season - Sentinel-2
- MSRM 1 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 60, x 2 [[7]](#7)[[2]](#2)
- MSRM 2 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 120, x 2 [[7]](#7)[[2]](#2)
- MSRM 3 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 240, x 2 [[7]](#7)[[2]](#2)
- MSRM 4 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 480, x 2 [[7]](#7)[[2]](#2)
- MSRM 5 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 960, x 2 [[7]](#7)[[2]](#2)
- MSRM 6 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 1920, x 2 [[7]](#7)[[2]](#2)
- MSRM 7 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 3840, x 2 [[7]](#7)[[2]](#2)
