# US Flight Delay Prediction

## 1. Project Introduction

### 1.1 Introduction
Flight delays remain a persistent challenge in the U.S. aviation industry, causing significant economic losses and passenger dissatisfaction. This project analyzes flight delay patterns using the Aeolus dataset, a comprehensive multi-modal dataset sourced from the U.S. Department of Transportation's Bureau of Transportation Statistics. The analysis focuses on domestic flights during December 2023 and December 2024, encompassing over 1.13 million flight records with rich operational, meteorological, and airport-level attributes.

The project has two main analytical components. First, exploratory data analysis is conducted to understand delay patterns, weather impacts, and year-over-year trends. Second, predictive models are developed for two complementary tasks: a regression task to predict exact arrival delay in minutes, and a classification task to predict whether a flight will experience a significant delay exceeding 15 minutes. All analyses are implemented using the tidymodels framework in R.

### 1.2 Problem Statement
Flight delays cost the U.S. economy billions of dollars annually, yet accurately predicting delays remains difficult due to the complex interaction of weather conditions, operational factors, airport congestion, and delay propagation across flight networks. During the December holiday travel season, the problem intensifies as flight volumes surge and weather disruptions become more frequent.

Existing research has rarely compared regression and classification approaches on a unified dataset with integrated weather features. Furthermore, delay pattern changes between consecutive holiday seasons remain underexplored. This project addresses these gaps by leveraging the Aeolus dataset to simultaneously investigate both predictive paradigms while focusing specifically on December 2023 and 2024.

### 1.3 Research Objectives
The objectives of this project are as follows:

1. To perform data cleaning and preprocessing on the Aeolus dataset for both regression and classification tasks.
2. To conduct exploratory data analysis characterizing flight delay patterns in December 2023 and December 2024.
3. To develop a regression model predicting exact arrival delay (minutes) using operational, temporal, meteorological, and geospatial features.
4. To develop a classification model predicting whether a flight will be delayed more than 15 minutes.
5. To identify the most influential predictors of flight delays through feature importance analysis.
6. To compare the performance of regression versus classification approaches for delay prediction.

### 1.4 Research Questionsn
1. To what extent can weather conditions (temperature, precipitation, wind speed) predict flight arrival delays during December?
2. How do flight delay patterns differ between December 2023 and December 2024?
3. Which features (operational, temporal, meteorological, or geospatial) are the strongest predictors of arrival delay?
4. How does a regression model compare to a classification model in predicting flight delays?
5. Can a binary classification model accurately identify flights delayed more than 15 minutes?

## 2. Dataset Description

### 2.1 Data Source
The dataset used in this project is titled Aeolus - A Multi-Modal Flight Delay Dataset. It originates from the U.S. Department of Transportation's Bureau of Transportation Statistics, which collects and publishes comprehensive data on domestic flight operations. The dataset covers domestic flights within the United States during the month of December for two consecutive years, specifically December 1–31, 2023 and December 1–31, 2024. This time period was selected to focus on the peak holiday travel season, allowing for comparative analysis of delay patterns between two holiday periods.

### 2.2 Dataset Dimension
The dataset comprises 1,133,043 rows and 35 columns. The breakdown by year is as follows:
| Year | Number of Flights |
|------|--------------------|
| December 2023 | 558,609 |
| December 2024 | 574,434 |
| Total | 1,133,043 |

### Variable Structure

The dataset contains 35 variables distributed across three data types:
| Data Type | Number of Columns |
|-----------|------------------|
| Datetime | 7 |
| Numeric | 24 |
| Character | 3 |

The variables are organized into six categories:

| Category | Variables |
|----------|-----------|
| Flight Identification | OP_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST |
| Temporal | FL_DATE, CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME, MONTH, DAY_OF_MONTH, DAY_OF_WEEK |
| Delay Metrics | DEP_DELAY, ARR_DELAY, TAXI_OUT, TAXI_IN, AIR_TIME |
| Weather (Origin) | O_TEMP, O_PRCP, O_WSPD |
| Weather (Destination) | D_TEMP, D_PRCP, D_WSPD |
| Geospatial | O_LATITUDE, O_LONGITUDE, D_LATITUDE, D_LONGITUDE |

## 3. Data Preparation

```r
# install dependencies 
install.packages(c(
  'tidyverse',
  'tidymodels',
  'janitor',
  'skimr',
  'glmnet',
  'ranger',
  'pROC',
  'rlang'
))

# load the libraries
library(tidyverse)
library(tidymodels)
library(janitor)
library(skimr)
library(glmnet)
library(ranger)
library(pROC)
library(rlang)
```
```r
# load dataset file paths
path_2023 <- 'flight_with_weather_2023.csv'
path_2024 <- 'flight_with_weather_2024.csv'

# read the data
flight_2023 <- read_csv(path_2023)
flight_2024 <- read_csv(path_2024)
```

File paths for the 2023 and 2024 datasets are first assigned to variables path_2023 and path_2024. The ```read_csv()``` function then reads each CSV file into R as a data frame. The output confirms that flight_2023 contains 6,645,461 rows and 34 columns, while flight_2024 contains 6,284,841 rows and 34 columns. 

```r
# inspect column names 2023
names(flight_2023)
```
**Output**
```r
[1] "FL_DATE"             "OP_CARRIER"          "OP_CARRIER_FL_NUM"  
 [4] "ORIGIN"              "DEST"                "CRS_DEP_TIME"       
 [7] "DEP_TIME"            "DEP_DELAY"           "TAXI_OUT"           
[10] "WHEELS_OFF"          "WHEELS_ON"           "TAXI_IN"            
[13] "CRS_ARR_TIME"        "ARR_TIME"            "ARR_DELAY"          
[16] "CRS_ELAPSED_TIME"    "ACTUAL_ELAPSED_TIME" "AIR_TIME"           
[19] "FLIGHTS"             "MONTH"               "DAY_OF_MONTH"       
[22] "DAY_OF_WEEK"         "ORIGIN_INDEX"        "DEST_INDEX"         
[25] "O_TEMP"              "O_PRCP"              "O_WSPD"             
[28] "D_TEMP"              "D_PRCP"              "D_WSPD"             
[31] "O_LATITUDE"          "O_LONGITUDE"         "D_LATITUDE"         
[34] "D_LONGITUDE"
```

```r
# inspect column names 2024
names(flight_2024)
```

**Output**
```r
[1] "FL_DATE"             "OP_CARRIER"          "OP_CARRIER_FL_NUM"  
 [4] "ORIGIN"              "DEST"                "CRS_DEP_TIME"       
 [7] "DEP_TIME"            "DEP_DELAY"           "TAXI_OUT"           
[10] "WHEELS_OFF"          "WHEELS_ON"           "TAXI_IN"            
[13] "CRS_ARR_TIME"        "ARR_TIME"            "ARR_DELAY"          
[16] "CRS_ELAPSED_TIME"    "ACTUAL_ELAPSED_TIME" "AIR_TIME"           
[19] "FLIGHTS"             "MONTH"               "DAY_OF_MONTH"       
[22] "DAY_OF_WEEK"         "ORIGIN_INDEX"        "DEST_INDEX"         
[25] "O_TEMP"              "O_PRCP"              "O_WSPD"             
[28] "D_TEMP"              "D_PRCP"              "D_WSPD"             
[31] "O_LATITUDE"          "O_LONGITUDE"         "D_LATITUDE"         
[34] "D_LONGITUDE"
```

```r
# verify both datasets have identical column names and order
identical(names(flight_2023), names(flight_2024))
```
**Output**
```r
[1] TRUE
```
Column name inspection reveals 34 variables organized into flight identification, temporal data, delay metrics, weather conditions (origin/destination), and geospatial coordinates. The ```identical()``` function confirms both datasets have identical column names and ordering, validating that row-binding can proceed without alignment issues.


```r
# check unique values for MONTH column 2023 and 2024
unique(flight_2023$MONTH)
unique(flight_2024$MONTH)
```
**Output**
```r
[1]  1  2  3  4  5  6  7  8  9 10 11 12
[1]  1  2  4  5  6  7  8  9 10 11 12
```

The ```unique()``` function was applied to the MONTH column of the 2023 dataset to identify all distinct month values present. The output shows months 1 through 12, indicating that the 2023 dataset contains flight records from all twelve months of the year (January to December). 

However, for yesr 2024, the output shows months 1, 2, and 4 through 12, but notably month 3 (March) is missing. This indicates that the 2024 dataset does not contain any flight records for March. The reason for this missing month could be due to data unavailability at the time of download or an intentional subset provided by the data source.

```r
# filter to only keep Dec rows 
dec_flight_2023 <- flight_2023 %>%
  filter(MONTH == 12) %>%
  mutate(YEAR = 2023L)

dec_flight_2024 <- flight_2024 %>%
  filter(MONTH == 12) %>%
  mutate(YEAR = 2024L)

unique(dec_flight_2023$MONTH)
unique(dec_flight_2024$MONTH)

cat("December 2023:", nrow(dec_flight_2023), "flights\n")
cat("December 2024:", nrow(dec_flight_2024), "flights\n")
```
**Output**
```r
[1] 12
[1] 12
December 2023: 558609 flights
December 2024: 574434 flights
```

December flights are isolated using ```filter(MONTH == 12)```. This serves two purposes: 
1. focusing the analysis on the peak holiday travel season, 
2. reducing computational overhead from approximately 6.6 million to 558,609 rows (2023) and 6.3 million to 574,434 rows (2024). 

A YEAR column is appended using mutate() to preserve temporal context for subsequent comparative analyses.


```r
df <- bind_rows(dec_flight_2023, dec_flight_2024) %>%
  janitor::clean_names()

glimpse(df)
```
The two December datasets are combined vertically using ```bind_rows()```. The ```janitor::clean_names()``` function standardizes all column names to snake_case (e.g., ```OP_CARRIER → op_carrier```), ensuring consistency and preventing naming errors. The final dataset contains 1,133,043 rows and 35 columns (34 original variables plus the added year column).

```r
# Comprehensive summary
skim(df)

# summary statistics by year
df %>%
  group_by(year) %>%
  summarise(
    n_flights        = n(),
    mean_arr_delay   = round(mean(arr_delay, na.rm = TRUE), 2),
    median_arr_delay = round(median(arr_delay, na.rm = TRUE), 2),
    delay_rate_pct   = round(mean(arr_delay > 15, na.rm = TRUE) * 100, 2),
    mean_dep_delay   = round(mean(dep_delay, na.rm = TRUE), 2),
    n_carriers       = n_distinct(op_carrier),
    n_airports       = n_distinct(origin),
    .groups = "drop"
  )
```
**Output**
```
# A tibble: 2 × 8
   year n_flights mean_arr_delay median_arr_delay delay_rate_pct mean_dep_delay
  <int>     <int>          <dbl>            <dbl>          <dbl>          <dbl>
1  2023    558609           0.43               -8           15.0           8.16
2  2024    574434           6.95               -6           20.4          13.3 
# ℹ 2 more variables: n_carriers <int>, n_airports <int>
```

The ```skim()``` function provides a comprehensive statistical summary, revealing no missing values in core flight variables and minimal missingness in weather fields (0.02% for origin, 0.07% for destination). The grouped summary compares December 2023 and 2024 performance. Key findings include a 2.8% increase in flight volume, a rise in mean arrival delay from 0.43 to 6.95 minutes, and an increase in the proportion of flights delayed over 15 minutes from 15.0% to 20.4%