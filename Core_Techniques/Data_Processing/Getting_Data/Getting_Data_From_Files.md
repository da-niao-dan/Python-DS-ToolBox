# Read Data From Files
```
import pandas as pd
```
## csv file
csv stands for comma-separated values. Our target is to import csv file into a PandasDataFrame.
The first step is to understands what is the delimiter of the file.
Then we use the function read_csv in pandas.

Suppose f is a csv file with tab as delimiters, then we do this:
```
df = pd.read_csv(f,delimiter ='\t')
```

## json files
json files are human readable, you can think of them as python dictionaries. We want to load them into pandas DataFrame.
The furst step is to view your data to get a sense.
The second step is to use a function called json_normalize.
```
from pandas.io.json import json_normalize

df = json_normalize(yourjsonfile,['key1', 'key2', ...], max_level=3)
```

