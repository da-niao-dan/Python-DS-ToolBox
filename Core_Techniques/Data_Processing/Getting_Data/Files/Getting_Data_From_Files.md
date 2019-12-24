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

