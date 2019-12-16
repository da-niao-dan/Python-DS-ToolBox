
## This document introduces how we can get data from web api and how to process json files
## Many APIs return files in json type, so we put these two topics together.

## Getting data from web API usually require reading API manuals for specific APIs.
## To use web API, learn basic http commands first.

## If you now understands http, now we continue.
## I recommend using package requests for this task in Python

import requests

### Specify url and additional query sentences according to your need and specific API mannuals.
url = 'http://....'
param_dict = ...
header = {'accept':'application/json',
          'Content-Type':'application/json' } 
with requests.Session() as s: 
    download = s.post(url,headers = header, data=json.dumps(param_dict))


## Here download is an instance of 'response' class defined in requests library. 
## Use reponse.json method to transform result into a dictionary
download_dict = download.json()


## Now, take a look at your json data pulled from api
## In this example, we got json files back.
## I recommend using Python library json to process json data.
import json
print(json.dumps(download_dict, indent=4, sort_keys=True))
## note json.dumps produces a json formatted string.

## Here requests.json() did the work of loading json to dictionary for us.
## if we get json string directly, we can use function loads from json package
json_str = "{}"
dict =json.loads(json_str)

## Sometimes we can convert dictionary into pandas dataframe by hand
## More often we need handier tool.

import pandas
from pandas.io.json import json_normalize

## I recommend using function json_normalize for this task
df = json_normalize(download.json(),['detail_result','rows'], max_level = 2)

