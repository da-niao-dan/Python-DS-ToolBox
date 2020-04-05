# Functions in Python
I recommend this excellent [course](https://campus.datacamp.com/courses/writing-efficient-code-with-pandas/), alongside with the officual document.

## Best Practices

### Docstrings

#### Google Style

```python
def function(ar_1,arg_2=42):
    """Imperative language description here
    

    Args:
      arg_1 (str): Description
      arg_2 (int,optional): Write optional when an argument has a 
        default value.



    Returns:
      bool: Optional description of the return value
      Extra lines are not indented.

    Raises:
      ValueError: Include any error types that the function 
        intentionally raises

    Notes:
      See https://www.website.com
      for more info.
    """
    return something


## retrieve the doc string
print(function.__doc__)

import inspect
print(inspect.getdoc(function))

```


