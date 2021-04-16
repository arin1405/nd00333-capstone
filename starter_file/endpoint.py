import requests
import json

# URL for the web service:
scoring_uri = 'http://d691f954-97bb-4f24-8893-1712876f9ba2.southcentralus.azurecontainer.io/score'
key = 'd3s5zbAZjRSpehuA3gt6hJmyVTd2Mfko'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "mean_radius": 7.75,
            "mean_texture": 24.64,
            "mean_perimeter": 47.94,
            "mean_area": 180.0,
            "mean_smoothness": 0.05264
          },
          {
            "mean_radius": 20.5,
            "mean_texture": 29.34,
            "mean_perimeter": 140.2,
            "mean_area": 1265.1,
            "mean_smoothness": 0.1177
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
