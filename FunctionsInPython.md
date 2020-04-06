# Functions in Python
I recommend this excellent [course](https://campus.datacamp.com/courses/writing-efficient-code-with-pandas/), alongside with the officual document.

Note that these notes are not as near clear as the explaination in that above course.

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

### Don't Repeat Yourself (DRY) and Do One Thing (DOT)
Do not repeat yourself, and each function should only do one thing.

### Pass by Assignment
Know which types are mutable and which are immutable.

Best Practices for defaulting on a mutable object:
is setting the default to None rather than [], {}, empty df etc..

```python
# Use an immutable variable for the default argument 
def better_add_column(values, df=None):
  """Add a column of `values` to a DataFrame `df`.
  The column will be named "col_<n>" where "n" is
  the numerical index of the column.

  Args:
    values (iterable): The values of the new column
    df (DataFrame, optional): The DataFrame to update.
      If no DataFrame is passed, one is created by default.

  Returns:
    DataFrame
  """
  # Update the function to create a default DataFrame
  if df is None:
    df = pandas.DataFrame()
  df['col_{}'.format(len(df.columns))] = values
  return df
```

## Context Manager

```python
with <context-manager>(<args>) as <variable-name>:
    # Run your code here
    # This code is running "inside the context"
# This code runs after the context is removed.    
```

### Writing context manager

* Class-based
* Function-bashed

#### Function based

```python
@contextlib.contextmanager
def my_context():
    # Add any set up code you need
    yield
    # Add any teardown code you need
```

```python
# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.

  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))

with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)
```

```python
@contextlib.contextmanager
def open_read_only(filename):
  """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
  read_only_file = open(filename, mode='r')
  # Yield read_only_file so it can be assigned to my_file
  yield read_only_file
  # Close read_only_file
  read_only_file.close()

with open_read_only('my_file.txt') as my_file:
  print(my_file.read())
```

#### Handling Errors
```python
try:
    # code that might raise an error
except:
    # do something about the error
finally:
    # this code runs no matter what
```

Be sure to clean up the mess after the party even if an error occured.

```python
def in_dir(directory):
  """Change current working directory to `directory`,
  allow the user to run some code, and change back.

  Args:
    directory (str): The path to a directory to work in.
  """
  current_dir = os.getcwd()
  os.chdir(directory)

  # Add code that lets you handle errors
  try:
    yield
  # Ensure the directory is reset,
  # whether there was an error or not
  finally:
    os.chdir(current_dir)
```

## Decorators

### Function as an object
Function is an object, like any other things in python.
```python
def my_function():
    print('Hello')
x = my_function
type(x)
x()


list_of_functions = [my_function, open, print]

list_of_functions[2]('I am printing with an element of a list!')

dict_of_functions = {
    'key1': my_function,
    'key2': open,
    'key3': print
}

dict_of_functions['func3']('I am printing with a value of a dict!')
```

```python
# Add the missing function references to the function map
function_map = {
  'mean': mean,
  'std': std,
  'minimum': minimum,
  'maximum': maximum
}

data = load_data()
print(data)

func_name = get_user_input()

# Call the chosen function and pass "data" as an argument
function_map[func_name](data)
```
### Scope

Local, Nonlocal, Global, Builtin Scope

global and nonlocal are keywords to access higher scope variables.

```python
x = 0
def my_func():
    global x
    x += 1
    print(x)
    y=2
    def func2():
        nonlocal y
	y+=1
	print(y)
```

### Closure
Nonlocal variables attached to a return function, so that the function can operate outside it parent scope.
```python
my_function.__closure__ # can access the closure

def return_a_func(arg1, arg2):
  def new_func():
    print('arg1 was {}'.format(arg1))
    print('arg2 was {}'.format(arg2))
  return new_func
    
my_func = return_a_func(2, 17)

print(my_func.__closure__ is not None)
print(len(my_func.__closure__) == 2)

# Get the values of the variables in the closure
closure_values = [
  my_func.__closure__[i].cell_contents for i in range(2)
]
print(closure_values == [2, 17])
```

### Decorators

```python
def print_before_and_after(func):
  def wrapper(*args):
    print('Before {}'.format(func.__name__))
    # Call the function being decorated with *args
    func(*args)
    print('After {}'.format(func.__name__))
  # Return the nested function
  return wrapper

@print_before_and_after
def multiply(a, b):
  print(a * b)

multiply(5, 10)
```

Time and Space are two of the most powerful... OK, seriously:

```python
import time

def timer(func):
    """A decorator that prints how long a function took to run."""

    # Define the wrapper function to return
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
	t_start = time.time()
	# Call the decorated function and store the result.
	result = func(*args, **kwargs)
	# Get the total time it took to run, and print it.
	t_total = time.time() - t_start
	print('{} tool {}s'.format(func.__name__,t_total))
	return result
    return wrapper

    
def memorize(func):
    """Store the results of the decorated function for fast lookup"""
    
    # Store results in a dict that maps arguments to results
    cache = {}

    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
	if (args,kwargs) not in cache:
	    # Call func() and store the result.
	    cache[(args,kwargs)] = func(*args, **kwargs)
	return cache[(arg,kwargs)]

    return wrapper
```

Example of a counter decorator:
```python
def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    # Call the function being decorated and return the result
    return func(*args,**kwargs)
  wrapper.count = 0
  # Return the new decorated function
  return wrapper

# Decorate foo() with the counter() decorator
@counter
def foo():
  print('calling foo()')
  
foo()
foo()

print('foo() was called {} times.'.format(foo.count))
```

### Decorators and metadata
Decoraters may mess up metadata of functions like function.__name__ or function.__doc__
There is a decorator to use when you are defining a decorator to fix this:

```python
from functools import wraps

def timer(func):
    """A decorator that prints how long a function took to run."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()

	result = func(*args, **kwargs)

	t_total = time.time() - t_start
	print('{} took {}s'.format(func.__name__, t_total))
    return wrapper

@timer
def sleep_n_seconds(n=10):
    """Pause processing for n seconds.
    
    Args:
      n (int): The number of seconds to pause for.
    
    """      
    time.sleep(n)

print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)


## More over you can get the original function back
sleep_n_seconds.__wrapped__

```

### Decorators that take arguments
In fact we just need to create a "function" that returns decorators, and call that "function" decorators as well.
Example of timeout:
```python
import signal

def raise_timeout(*args,**kwards):
    raise TimeoutError()

# When an "alarm" signal goes off, call raise_timeout()
signal.signal(signalnum=signal.SIGALRM, handler=raise_timeout)

# Set off an alarm in 5 seconds
signal.alarm(5)

# cancel the alarm
signal.alarm(0)


def timeout(n_seconds):
    def decorator(func):
        @wraps(func)
	def wrapper(*args, **kwargs):
            # Set an alarm for n seconds
	    signal.alarm(n_seconds)

	    try:
	        # Call the decorated func
		return func(*args,**kwargs)
            finally:
	        # Cancel alarm
		signal.alarm(0)
        return wrapper
    return decorator
```

Another example:
```python
def tag(*tags):
  # Define a new decorator, named "decorator", to return
  def decorator(func):
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
      # Call the function being decorated and return the result
      return func(*args, **kwargs)
    wrapper.tags = tags
    return wrapper
  # Return the new decorator
  return decorator

@tag('test', 'this is a tag')
def foo():
  pass

print(foo.tags)
```

Debug tool: type checking:
```python
def returns(return_type):
  # Complete the returns() decorator
  def decorator(func):
    def wrapper(*args):
      result = func(*args)
      assert(type(result) == return_type)
      return result
    return wrapper 
  return decorator
  
@returns(dict)
def foo(value):
  return value

try:
  print(foo([1,2,3]))
except AssertionError:
  print('foo() did not return a dict!')

```


