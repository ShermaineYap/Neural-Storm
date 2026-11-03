### The problem

APIMS shows Malaysia's Air Pollutant Index for the hour that just passed. During a haze season, schools, outdoor workers and people with asthma need to know about tomorrow, not the last hour. HazeCast MY gives each station a 24- and 48-hour forecast, a chance of Unhealthy air (API > 100), an 80% range, and the reason behind the number.

### Data

- Hourly API readings from 67 DOE continuous monitoring stations, from January 2013 to May 2022. That is 3.8 million station-hours, from data.gov.my (CC BY) and the public APIMS table, via the ynshung/api-malaysia archive.
- Only readings available at the time of the forecast are used.

### What the model sees (36 features per station-hour)

- **The station itself:** the current API; readings 1 h to 7 days ago; 3-, 6- and 24-hour changes; 6/24/72-hour averages; the 24-hour max and spread.
- **Its three nearest stations:** their average, maximum and 6-hour trend.
- **Its region** (Klang Valley, Sarawak, Peninsular South…): the regional mean and its 24-hour change.
- **The whole country:** the median, the 90th percentile, and the share of stations above 70 and above 100. Haze that blows in from outside Malaysia lifts all of these together.
- **Calendar:** hour, day of week, month, and whether it is the Southwest-monsoon haze season (Jun–Sep).

### Models

| Purpose | Model |
|---|---|
| API in 24 h and 48 h | LightGBM, which predicts the *change* from the current reading |
| Chance of Unhealthy air in the next 24 h | LightGBM classifier |
| 80% range | Split-conformal intervals, calibrated separately for predicted API < 50, 50–100 and > 100 |
| Why | Exact TreeSHAP contributions, grouped into driver families |

These are compared against persistence, 24-hour-mean persistence, climatology, ridge regression, a small neural network (MLP), and a common rule of thumb ("warn when API ≥ 80").

### Validation

Walk-forward only. Each model trains on earlier years and is tested on a later period it has never seen:

- 2013–2018 → 2019 (the severe September 2019 haze)
- 2013–2019 → 2020
- 2013–2020 → 2021 to May 2022

The last 15% of each training window is held back for early stopping and to calibrate the intervals. The alert threshold of 20% is set in advance from a cost ratio (missing an Unhealthy day counts 4× a false alarm) and is never tuned on test data.

### HazeAgent

For every station, the agent makes one of four decisions:

- **ALERT:** the chance of Unhealthy air is at least 20%.
- **WATCH:** the chance is lower, but tomorrow's 80% range reaches 100.
- **CLEAR:** neither of the above.
- **INSUFFICIENT DATA:** more than half of the last 24 hours is missing, so the agent declines to forecast rather than guess.

The agent also labels the episode. When neighbouring, regional and national signals push the forecast up more than the station's own history does, it reads as a **widespread (transboundary-haze) signature**. Otherwise it reads as **local build-up**.

### Limitations

- The archive holds only the composite API, not the separate PM2.5, PM10, O₃, CO, NO₂ and SO₂ readings. "Attribution" here therefore means *which signals drive the forecast*. It is not chemical source apportionment.
- No weather or fire-hotspot inputs are used (wind, rain, MODIS/VIIRS hotspots). Adding them is the clearest next step.
- The public archive ends in May 2022, so the app replays history rather than serving a live feed.
- Station coordinates are approximate town centres.
- This is an experimental research forecast. Always follow official DOE / APIMS advisories.
