import argparse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pandas as pd
import requests

def validate_number_list(data) -> list[int]:
    if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
        raise TypeError("Expected list of integers")
    return data

def get_observations(api_host: str, label: str) -> list[int]:
    url = f"{api_host}/{label}"
    response = requests.get(url)
    response.raise_for_status()
    data = validate_number_list(response.json())
    return data

def post_observation(api_host: str, label: str, value: int) -> None:
    url = f"{api_host}/{label}/{value}"
    response = requests.post(url)
    response.raise_for_status()

def main() -> None:
    parser = argparse.ArgumentParser(description="SARIMAX Forecasting")
    parser.add_argument("--api-host", required=True, help="API host URL")
    parser.add_argument("--forecast-days", type=int, required=True, help="Number of days to forecast")
    parser.add_argument("--records-per-day", type=int, required=True, help="Number of records per day")
    parser.add_argument("--input-label", required=True, help="Input label for observations")
    args = parser.parse_args()
    api_host = args.api_host.rstrip("/")
    forecast_days = args.forecast_days
    records_per_day = args.records_per_day
    input_label = args.input_label
    output_label = f"{input_label}.forecast-{forecast_days}d"

    data = pd.Series(get_observations(api_host, input_label))
    model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,records_per_day))
    fit: SARIMAXResults = model.fit()
    forecast = fit.forecast(steps=forecast_days * records_per_day)

    last_value = int(round(forecast.iloc[-1]))
    post_observation(api_host, output_label, last_value)

if __name__ == "__main__":
    main()