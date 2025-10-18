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

# References
<a id="1">[1]</a> 
Orengo, Hector A., and Cameron A. Petrie. (2017). 
Large-scale, multi-temporal remote sensing of palaeo-river networks: A case study from Northwest India and its implications for the Indus civilisation.
Remote Sensing, 9(7), 735.
http://www.mdpi.com/2072-4292/9/7/735
https://github.com/horengo/Orengo_Petrie_2017_RS

<a id="2">[2]</a> 
Buławka, Nazarij, and Hector A. Orengo. (2024).
Application of multi-temporal and multisource satellite imagery in the study of irrigated landscapes in arid climates.
Remote Sensing, 16(11), 1997.
https://doi.org/10.3390/rs16111997

<a id="3">[3]</a> 
Garcia-Molsosa, Arnau, Hector A. Orengo, and Cameron A. Petrie. (2023).
Reconstructing long-term settlement histories on complex alluvial floodplains by integrating historical map analysis and remote-sensing: an archaeological analysis of the landscape of the Indus River Basin.
Heritage Science, 11(1), 141.
https://heritagesciencejournal.springeropen.com/articles/10.1186/s40494-023-00985-6

<a id="4">[4]</a> 
Jin, Suming, and Steven A. Sader. (2005). 
Comparison of Time Series Tasseled Cap Wetness and the Normalized Difference Moisture Index in Detecting Forest Disturbances.
Remote Sensing of Environment, 94(3), 364–72. 
https://doi.org/10.1016/j.rse.2004.10.012.

<a id="5">[5]</a> 
Gao, Bo-cai. (1996). 
NDWI—A Normalized Difference Water Index for Remote Sensing of Vegetation Liquid Water from Space.
Remote Sensing of Environment, 58(3), 257–66. 
https://doi.org/10.1016/S0034-4257(96)00067-3.

<a id="6">[6]</a> 
Ji, Lei, Li Zhang, and Bruce Wylie. (2009). 
Analysis of Dynamic Thresholds for the Normalized Difference Water Index. 
Photogrammetric Engineering & Remote Sensing 75(11): 1307–17. 
https://doi.org/10.14358/PERS.75.11.1307.

<a id="7">[7]</a> 
Orengo, Hector A., and Cameron A. Petrie. (2018). 
Multi‐scale Relief Model (MSRM): A New Algorithm for the Visualization of Subtle Topographic Change of Variable Size in Digital Elevation Models/
Earth Surface Processes and Landforms, 43(6), 1361–69. 
https://doi.org/10.1002/esp.4317.
https://github.com/horengo/Orengo_Petrie_2018_MSRM.
