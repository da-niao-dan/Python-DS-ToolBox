# Get data from web
Before we begin, have some basic knowledge on how the internet works and what is http, and what is API.
Getting data from web API usually require reading API manuals for specific APIs.
I recommend using package requests for this task in Python.

```
# Assign URL to variable: url

url = 'yoururl' # Consult your api documents here.

header = {"Accept-Encoding":"gzip"} # Consult your api documents and requests package website here.

# Package the request, send the request and catch the response: r
param_dict = {your_parameters}# Consult your api documents and requests package website here.

with requests.Session() as s: 
    download = s.post(url, data=param_dict) ## consult post, get, and other http requests.
```    
