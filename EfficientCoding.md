# Pythonic Code!
We want to write fast, memory efficient and super readable code.
Try this out:
```python
import this
```

At the beginning of my career, I did not pay a lot attention to this.
However, you really need this not just for efficiency but also for being able to understand pythonic code.


## Introduction

### Unpacking and list apprehension
Beyond List Apprehension: Unpacking!
Convert a range object into a list:
```python
my_list = [*range(1,99,2)]
```

Use enumerate for index and value pairs:
```python
indexed_vals_comp = [(i,val) for i, val in enumerate(vals)]
## But this is not pythonic enough! Do this:
indexed_vals_unpack = [*enumerate(vals, 1)] ## Begin index at 1 

## try unpack with map()

verbs_map = map(str.upper,verbs)
verb_uppercase =[*verbs_map]
```

Summerizing: you only need list apprehension if you are doing not vectorizble actions to entries in a iterator.

### Numpy 
Boolean Indexing and Broadcasting
Re-bind the IDs and the features
```python
old_features = [*range(2,99,3)]

old_features_np = np.array(old_features)
new_features = old_features * 9

new_bindings = [(IDs[i], new_feature) for i,new_feature in enumerate(new_features)]

## Use the new bindings for new mappings
def phrase(index,i):
    return " ".join([index,"record has val",i])

phrase_map = map(phrase,new_bindings)

phrases = [*phrase_map]
print(*phrases,sep='\n')
```

## Examing Run Time
IPython Magic commands are enhancements on top of normal Python syntax
Prefixed by the "%" character

Try this out:
```python
%lsmagic
```

Now we focus on %timeit
```python
%timeit

%%timeit
```
After these, you can discover which codes are more efficient than others!

### runs and times per run
```python
%timeit -r9 -n12 formal_dict = dict()
%timeit -r9 -n12 literal_dict = {}
```

### cell magic
```python
%%timeit
## Anycode in this cell...
```
### actual timings
```python
times = %timeit -o set(my_stuff)

## times for each run
print(times.timings)
print(times.best)
print(times.worst)
print(times.average)

```
## Code profiling

### Profiling time consumption
Let's see the frequency and timing of each lines of a code
```python
!pip install line_profiler

%load_ext line_profiler

## Magic command for line-by-line times
%lprun

## profile a function
%lprun -f function_name function_name(arg1,arg2,agr3)

```

### Profiling memory consumption

```python
!pip install memory_profiler

%load_ext memory_profiler

## Magic command
%mprun

## profile a function
%mprun -f function_name function_name(arg1,arg2,agr3)

```

## Efficient Functions

### zip

```python
new_list = [*zip(l1,l2,l3,...)] ## Note that each li is an iterable, a list can be combined with a, for example, map object.
```

### Counter

```python

from collections import Counter

# Use list comprehension to get each names's starting letter
starting_letters = [a[0] for a in names]

# Collect the count of names for each starting_letter
starting_letters_count = Counter(starting_letters)
```


### itertools

#### combinations
```python
from itertools import combinations

list_tup = [*combinations(obj_list,num)] ## choose num from obj_list

```

### set
set methods
* itersection()
* difference() # a.difference(b) = a\b
* symmetric_difference() # all elements in exactly one list
* union()

Use built-in function for membership testing:
* in

### Eliminating Loops
Using np vectorizations, map, itertools.

### Writing more efficient Loops

Move operations outside loop as much as possible:
```python
# Collect all possible pairs using combinations()
possible_pairs = [*combinations(types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

# Add a line to append each enumerated_pair_tuple to the empty list above
for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)
```


## Iterrating over Pandas Dataframe
```python
for i,row in df.iterrows():
    print(i,row)
```
This is much faster than for loop with iloc

However, there is even a faster approach:

```python
for row_namedtuple in team_wins_df.itertuples():
    print(row_namedtuple.Team)
```

If we want to apply a function for each column (0) or row (1), use df.apply.

### Pandas internals

Even better than apply!!!

Vectorize Operations!

```python
%%timeit
win_perc_preds_loop = []

# Use a loop and .itertuples() to collect each row's predicted win percentage
for row in baseball_df.itertuples():
    runs_scored = row.RS
    runs_allowed = row.RA
    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)
    win_perc_preds_loop.append(win_perc_pred)

%%timeit
# Apply predict_win_perc to each row of the DataFrame
win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)

%%timeit
# Calculate the win percentage predictions using NumPy arrays
win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)
baseball_df['WP_preds'] = win_perc_preds_np

print(baseball_df.head())
```


