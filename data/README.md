# Data

`apims-2005-2017.csv` and `apims-2018-now.csv` are the hourly Air Pollutant Index (API)
readings for Malaysia's DOE continuous air-quality monitoring stations, one column per
station, one row per hour (2005-10-01 to 2022-05-07).

Source: https://github.com/ynshung/api-malaysia (file `api-malaysia.zip`).
2005-2017 originates from data.gov.my (MOSTI / DOE / KeTSA), licensed CC BY (Open Definition 2.1).
2018 onwards was scraped from the public APIMS table (http://apims.doe.gov.my).

HazeCast uses 2013-01 onwards, where readings are hourly; earlier years have only
2-3 readings per day. 2015-2016 have large gaps in the archive.
