# Hydrograph Goodness-of-Fit Web App

This app evaluates goodness-of-fit metrics for simulated and observed hydrographs over a user-selected time interval.

## Input Data

1. Extract the following ordinates from the design/modeling software:
   - `DATE`
   - `TIME`
   - `SIM` (simulated flow)
   - `OBS` (observed flow)

2. Save the extracted data as a `.csv` file.

3. The CSV must follow the same format as the sample file:

   `CONT_SIM_TIMESERIES.csv`

## Running the App

1. Download the Python script (`.py`) to your local computer.
2. Open the script using a Python compiler or IDE.
3. Run the script locally to launch the web app.

## Notes

- The app requires the input data to be in the exact format shown in the sample CSV file.
- The goodness-of-fit metric formulas used in the app are included as comments inside the script.
- The Date should be in DDMMMYYY format and Time should be in HH:MM format
