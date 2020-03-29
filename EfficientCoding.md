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
Let's see the frequency and timing of each lines of a code
```python
!pip install line_profiler

%load_ext line_profiler

## Magic command for line-by-line times
%lprun

## profile a function
%lprun -f function_name function_name(arg1,arg2,agr3)

```


