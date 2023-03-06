import requests
import json
from datetime import date, timedelta

# Define the location and date for which to retrieve weather data
location_id = '310011'  # Location ID for Leicester, can be found on DataPoint website
start_date = date.today() + timedelta(days=365) # a year from today
end_date = start_date + timedelta(days=1)

# Define the API endpoint and parameters
api_endpoint = 'http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/'
api_params = {'key': '50eda44e-179e-4ecf-81a3-7616e75b5cf3', 'res': '3hourly', 'time': '/' + start_date.isoformat() + 'Z' + '/' + end_date.isoformat() + 'Z'}

# Make the API request
response = requests.get(api_endpoint + location_id, params=api_params)

# Parse the response as JSON
print(response.content)
data = json.loads(response.content)

# Print the weather data
for forecast in data['SiteRep']['DV']['Location']['Period']:
    print('Date:', forecast['value'])
    for hour in forecast['Rep']:
        print('Time:', hour['$'])
        print('Temperature (C):', hour['T'])
        print('Weather Type:', hour['W'])
