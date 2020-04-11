# Creating Robust Workflows in Python
Notes are from this [course](https://learn.datacamp.com/courses/creating-robust-workflows-in-python).

Objectives: learn to:
* Follow best practices
* Use helpful technologies
* Write Python code that is easy to
    * read
    * use
    * maintain
    * share
* Develop your own workflow 
    * Steps you take
    * Tools you use
    * can change


## Python Programming Principles

###  Don't Repeat Yourself
The opposite of DRY is WET (Waste Everyone's Time).
Simply write repetive functions is not DRY. You may repetively call functions.
Example of a DRY code:

```python

def read(filename):
    with open(filename,'r') as file:
        return file.read()


# Use read() to read text files
filenames = ["diabetes.txt","boston.txt","iris.txt"]
# Read files with a list comprehension
diabetes, boston, iris = [read(f) for f in filenames]
```

However, we can be DRYer by using Standard Library
```python
from pathlib import Path

# Create a list of filenames
filenames = ["diabetes.txt","boston.txt","iris.txt"]

# Use pathlib in a list comprehension
diabetes, boston, iris = [Path(f).read_text() for f in filenames] ## Use Standard Library whenever possible
```

Moreover, DRY is not only about reability optimization, but also about time and memory optimization.
Turn a list comprehension into a generator expression to save memory.

```python
from pathlib import Path

# Create a list of filenames
filenames = ["diabetes.txt","boston.txt","iris.txt"]

# Use pathlib in a generator expression, just replace [] with ()
diabetes, boston, iris = (Path(f).read_text() for f in filenames) ## Use Standard Library whenever possible
```

Examples:
```python
matches = [['  :Number of Instances: 442\n',
  '  :Number of Attributes: First 10 columns are numeric predictive values\n'],
 ['    :Number of Instances: 506 \n',
  '    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n'],
 ['    :Number of Instances: 5620\n', '    :Number of Attributes: 64\n'],
 ['    :Number of Instances: 150 (50 in each of three classes)\n',
  '    :Number of Attributes: 4 numeric, predictive attributes and the class\n'],
 ['    :Number of Instances: 178 (50 in each of three classes)\n',
  '    :Number of Attributes: 13 numeric, predictive attributes and the class\n']]
 
def flatten(nested_list):
    return (item 
            # Obtain each list from the list of lists
            for sublist in nested_list
            # Obtain each element from each individual list
            for item in sublist)

number_generator = (int(substring) for string in flatten(matches)
                    for substring in string.split() if substring.isdigit()) 
pprint(dict(zip(filenames, zip(number_generator, number_generator))))   ## pretty print !!!
## print filename: (numInstances,numAttributes)

```

### Modularity
Modules vs scripts
Modules:
* are imported
* provide tools
* def functions

Scripts:
* are run
* perform actions
* call functions

'__name__' pronounced dunder name

One function on purpose.

```python
def obtain_words(string):
    # Replace non-alphabetic characters with spaces
    return "".join(char if char.isalpha() else " " for char in string).split()

def filter_words(words, minimum_length=3):
    # Remove words shorter than 3 characters
    return [word for word in words if len(word) >= minimum_length]
if __name__ == '__main__':
    words = obtain_words(Path("diabetes.txt").read_text().lower())
    filtered_words = filter_words(words)
    pprint(filtered_words)
```

```python
def count_words(word_list):
    # Count the words in the input list
    return {word: word_list.count(word) for word in word_list}


if __name__ == '__main__':
    # Create the dictionary of words and word counts
    word_count_dictionary = count_words(filtered_words)

    (pd.DataFrame(word_count_dictionary.items())
    .sort_values(by=1, ascending=False)
    .head()
    .plot(x=0, kind="barh", xticks=range(5), legend=False)
    .set_ylabel("")
    )
    plt.show()
```

### Abstraction (OOP)
Principles:
* Hide implementation details
* Design user interfaces
* Facilitate code use

Tricks:
* Chaining use instance methods that return instances.
* use '@classmethod' decorator to write class methods.

```python
# ScikitData is a class in memory
# Fill in the first parameter in the pair_plot() definition
def pair_plot(self, vars=range(3), hue=None):
    return pairplot(pd.DataFrame(self.data), vars=vars, hue=hue, kind="reg")

ScikitData.pair_plot = pair_plot

# Create the diabetes instance of the ScikitData class
diabetes = ScikitData("diabetes")
diabetes.pair_plot(vars=range(2, 6), hue=1)._legend.remove()
plt.show()
```

```python
# Fill in the decorator for the get_generator() definition
@classmethod
# Add the first parameter to the get_generator() definition
def get_generator(cls, dataset_names):
    return map(cls, dataset_names)

ScikitData.get_generator = get_generator
dataset_generator = ScikitData.get_generator(["diabetes", "iris"])
for dataset in dataset_generator:
    dataset.pair_plot()
    plt.show()

```

Note that pairplot is a technique for visualization in exploratory data analysis.
Note how instance methods and classmethods enable fast plotting and hiding of the details.

## Documentation and Types

### Type check
Add type hints
Type checker setup:
* mypy type checker
* pytest testing framework
* pytest-mypy pytest plugin
```python
!pip install pytest mypy pytest-mypy
```
Prepare a file pytest.ini with the following:
```
[pytest]
addopts = --doctest-modules --mypy --mypy-ignore-missing-imports
```

Then
```python
!pytest my_function.py
```

```python
from typing import List

def cook_foods(raw_foods: List[str]) -> List[str]:
    return [food.replace('raw','cooked') for food in raw_foods]

```

```python
from typing import Optional

def str_or_none(optional_string: Optional[str] = None) -> Optional[str]:
    return optional_string

```

### Docstring
Docstring types
```python
"""MODULE DOCSTRING"""

def double(n: float) -> float:
    """ Multiply a number by 2 """
    return n * 2

class DoubleN:
    """The summary of what the class does.
    Arguments:
        n: A float that will be doubled.
    Attributes:
        n_doubled: A float that is the result of doubling n.
    Examples:
        >>> double(2.0)
	4.0
    """
    def __init__(self, n:float):
	self.n_doubled = n * 2


help(DoubleN) ## class docstring

```

Location determines the type:
* In definitions of 
    * Functions
    * Classes
    * Methods
* At the top of .py files
    * Modules
    * Scripts
    * __init__.py (This docstring is for entire directory/package)

#### Show docstring
Package Docstring
```python
import pandas
help(pandas)
```

Module docstring
```python
import double
help(double)
```

#### Test docstring examples
Same as before:
Example:
```python
!pytest double.py
```

More examples:
```python
def get_matches(word_list: List[str], query:str) -> List[str]:
    ("Find lines containing the query string.\nExamples:\n\t"
     ">>> get_matches(['a', 'list', 'of', 'words'], 's')\n\t"
     "['list', 'words']")
    return [line for line in word_list if query in line]

help(get_matches)

def obtain_words(string: str) -> List[str]:
    ("Get the top words in a word list.\nExamples:\n\t"
     ">>> from this import s\n\t>>> from codecs import decode\n\t"
     ">>> obtain_words(decode(s, encoding='rot13'))[:4]\n\t"
     "['The', 'Zen', 'of', 'Python']") 
    return ''.join(char if char.isalpha() else ' ' for char in string).split()
  
help(obtain_words)
```

### Reports
View changes made to notebooks
```python
! diff -c old.ipynb new.ipynb
## Better Approach: nbdime nbdiff command
! nbdiff old.ipynb new.ipynb
```

Notebook workflow package
1. Use nbformat to create notebooks from:
    * Markdown files (.md)
    * Code files (.py)
2. Use nbconvert to convert notebooks

#### nb object
```python
from nbformat.v4 import (new_notebook,new_code_cell,new_markdown_cell)
import nbformat

nb = new_notebook()
nb.cells.append(new_code_cell('1+1'))
nb.cells.append(new_markdown_cell('Hi'))
nbformat.write(nb,"mynotebook.ipynb")

# convert nb
## Option 1
from nbconvert.exporters import HTMLExporter
html_exporter = HTMLExporter()

## Option 2
from nbconvert.exporters import get_exporter
html_exporter = get_exporter('html')()

## Create an HTML report from a Jupyter notebook
contents = html_exporter.from_filename('mynotebook.ipynb')[0]

from pathlib import Path

Path('myreport.html').write_text(contents)
```

Build a function that build notebooks.

```python
def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    nb = new_notebook()
    nb.cells = [
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text()) 
        if name.endswith(".py")
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    ]
    return nb
    
pprint(nbuild(["intro.md", "plot.py", "discussion.md"]))
```

Build a function that converts notebooks to other formats:
```python
def nbconv(nb_name: str, exporter: str = "script") -> str:
    """Convert a notebook into various formats using different exporters."""
    # Instantiate the specified exporter class
    exp = get_exporter(exporter)()
    # Return the converted file"s contents string 
    return exp.from_filename(nb_name)[0]
    
pprint(nbconv(nb_name="mynotebook.ipynb", exporter="html"))
```

### Pytest tests
```python
import pytest

def test_addition():
    assert 1 + 2 == 3

@pytest.mark.parametrize('n',[0,2,4])
def test_even(n):
    assert n%2 ==0

## Confirm expected error
def test_assert():
    with pytest.raises(AssertionError):
        assert 1+2 == 4
```
#### Test-Driven Development

1. Write a function with docstring but no code block
2. Write a test
3. Run the failing test
4. Work on the module until it passes
5. Improve the code/functionality: go to step 2 

1

```python
def double(n: float) -> float:
    """Multiply a number by 2"""
```

2

```python
from double import double

def test_double():
    assert double(2) == 4
```

3

```python
!pytest test_double.py
```

4

```python
def double(n: float) -> float:
    """Multiply a number by 2"""
    return n * 2
```
5 ->2
```python
import pytest
from double import double

def test_raises():
    with pytest.raises(TypeError):
        double('2') 

```

6(in fact 3) In bash,

```bash

pytest test_raises.py

```

7(in fact 4 again)

```python

def double(n: float) -> float:
    """Multiply a number by 2"""
    return n * 2.

```


Example test for nbuild:

```python
@pytest.mark.parametrize("inputs", ["intro.md", "plot.py", "discussion.md"])
def test_nbuild(inputs):
    assert nbuild([inputs]).cells[0].source == Path(inputs).read_text()

show_test_output(test_nbuild)

```

Example test for nbconv:

```python

@pytest.mark.parametrize("not_exporters", ["htm", "ipython", "markup"])
def test_nbconv(not_exporters):
    with pytest.raises(ValueError):
        nbconv(nb_name="mynotebook.ipynb", exporter=not_exporters)

show_test_output(test_nbconv)

```

#### Project Organization

Each modules has a coupled test module, lives in test folder

* Test files (e.g. test_mymodule.py)
    * Kept in tests/ directory
* Modules (e.g. double.py)
    * Kept in packages (package is a folder with __init__.py file)
    * Along with __init__.py
* Project documentation generation, using [this](https://www.sphinx-doc.org/en/master/).

## Shell superpowers

two command-line interfaces

* argparse
* docopt

To use argparse:

* instantiate the ArgumentParser class
* Call the add_argument() method
* Call the parse_args() method
* Arguments are Namespace attributes

```python

parser = argparse.ArgumentParser()
parser.add_argument('ARGUMENT_NAME')
namespace = parser.parse_args()
namespace.ARGUMENT_NAME

```

To use docopt:

* Setup CLI in file docstring
* Pass __doc__ to docopt()
* Arguments are dictionary values


```python

"""Pass a single argument to FILENAME.
Usage: FILENAME.py ARGUMENT_NAME"""

argument_dict = docopt.docopt(__doc__)

```

Examples:
Using argparse
get_namespace.py:

```python

import argparse as ap
def get_namespace() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument('format')
    return parser.parse_args()

```

now.py:

```python
import datetime
from get_namespace import get_namespace
if __name__ = '__main__':
    namespace = get_namespace()
    print(datetime.datetime.now()
    	  .strftime(namespace.format))
```

shell:

```bash
python now.py %H:%M
```

Using Docopt and docstrings:

get_arg_dict.py:

```python
"""Get the current date or time.
Usage: now.py FORMAT"""
from typing import Dict
import docopt
def get_arg_dict() -> Dict[str, str]:
    return docopt.docopt(__doc__)
```

now.py:

```python
import datetime as dt
from get_arg_dict import get_arg_dict
if __name__ == '__main__':
    arg_dict = get_arg_dict()
    print(dt.datetime.now()
          .strftime(arg_dict['FORMAT']))

```

Shell:

```bash

python now.py %B/%d

```

### GNU Make
automate execution of shell commands and scripts
and python scripts and modules.

Write recipes in Makefile:

```bash

TARGET:
    RECIPE

TARGET:
    RECIPE

TARGET: DEPENDENCIES

```

Example of a Maefile:

```bash

time:
    python now.py %H:%M

date:
    python now.py %m-%d

all: date time

```

### Building a CLI!

Build a argparse CLI for nbuild function

```python

def argparse_cli(func: Callable) -> None:
    # Instantiate the parser object
    parser = argparse.ArgumentParser()
    # Add an argument called in_files to the parser object
    parser.add_argument("in_files", nargs="*")
    args = parser.parse_args()
    print(func(args.in_files))

if __name__ == "__main__":
    argparse_cli(nbuild)

```

docopt:

```python

# Add the section title in the docstring below
"""Usage: docopt_cli.py [IN_FILES...]"""

def docopt_cli(func: Callable) -> None:
    # Assign the shell arguments to "args"
    args = docopt(__doc__)
    print(func(args["IN_FILES"]))

if __name__ == "__main__":
    docopt_cli(nbuild)

```

### GitPython API

initialze a git directory

init.py:

```python

import git
print(git.Repo.init())

```


```bash

python init.py

```

Add untracked files and commit:
commit.py:

```python

import git

repo = git.Repo()
add_list = repo.untracked_files

if add_list:
    repo.index.add(add_list)
    new = f"New files: {','.join(add_list)}."
    print(repo.index.commit(new).message)

```

bash:

```bash

python commit.py

```

Commit modified files:
mcommit.py:

```python

import git
repo = git.Repo()
diff = repo.index.diff(None).iter_change_type('M')
edit_list = [file.a_path for file in diff]
if edit_list:
    repo.index.add(edit_list)
    modified = f"Modified files: {', '.join(edit_list)}."
    print(repo.index.commit(modified).message)

```




Automate version control by adding a Makefile:

```bash

.git/:
    python init.py
    python commit.py
    python mcommit.py

```

all need to do later is to type in bash shell:

```bash

make

```
### Virtual Environments

venv is the python default environment manager, so is the most stable one.

#### Create environment in bash:

```bash
python -m venv /path/to/new/virtual/environment

## Option1: Activate environment before pip install
source /path/to/new/virtual/environment/bin/activate
pip install --requirement requirements.txt

## Option2L Use path to virtual environment Python to pip install
/path/to/new/virtual/environment/bin/python -m pip install -r requirements.txt
```

#### Make venv Example: 

1. Create a virtual environment in .venv
2. Install the dependencies in the virtual environment
3. Update the "Last Modified" time for the target(.vev/bin/activate)
4. test the project using the virtual env

```bash

.venv:
    python -m venv .venv

.venv/bin/activate: .venv requirements.txt
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/bin/activate

test: .venv/bin/activate
    .venv/bin/python -m pytest src tests

```


#### Create virtual environment in python:

```python
import venv
import subprocess

venv.create('.venv')
cp = subprocess.run(
    ['.venv/bin/python', '-m', 'pip', 'list'], stdout=-1
)

print(cp.stdout.decode())
```

More examples:

```python
# Create an virtual environment
venv.create(".venv")

# Run pip list and obtain a CompletedProcess instance
cp = subprocess.run([".venv/bin/python", "-m", "pip", "list"], stdout=-1)

for line in cp.stdout.decode().split("\n"):
    if "pandas" in line:
        print(line)

```

```python

print(run(
    # Install project dependencies
    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
    stdout=-1
).stdout.decode())

print(run(
    # Show information on the aardvark package
    [".venv/bin/python", "-m", "pip", "install", "aardvark"], stdout=-1
).stdout.decode())

```


### Persistence and Packaging

#### Persistence in a pipeline

Save files throughout a data analysis pipeline

raw data -> wrangle -> final data -> plot & model -> plots and trained models

Each file should be:

* A target in the project's Makefile
* The responsibility of a script (plot1.py -> plot1.png)

#### Saving data

Standardard pickle package is always a choice. Check it out.

With pandas:
```python
# Make simple dataframe
df = pd.DataFrame({
    'Evens': range(0,5,2),
    'Odds': range(1,6,2)

})

# Pickle dataframe
df.to_pickle('numbers.pkl')

# Unpickle dataframe
pd.read_pickle('numbers.pkl')
```

With joblib package

```python

import joblib
from sklearn import datasets, model_selection, neighbors

diabetes = datasets.load_diabetes()

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    diabetes.data, diabetes.target, test_size = 0.33, random_state=42
)

knn = neighbors.KNeighborsRegressor().fit(x_train, y_train)

# Pickle the k-nearest neighbots model with joblib
joblib.dump(knn, 'knn.pkl')

knn = joblib.load('knn.pkl')

```

#### Package

* Everything in one place:
   
   * code
   * documentation
   * data files
* Easy to share:
    * pip install mypkg (download from PyPI)
    * pip install git+REPO_URL (install from git repository)
    * pip install . (install local package)

To Create a package we need a setup.py file:
```python

setuptools.setup(
    name = "PACKAGE_NAME",
    version="MAJOR.MINOR.PATCH",
    description="A minimal example of packaging pickled data and models.",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    # Include files in the data and models directories
    package_data={"":["data/*","models/*"]},
)

```

To install the local package we add the following line to requrements.txt file and install:
```bash
--editable .
```

##### Upload a package

* create a PyPI account
* write a make file

release
```bash

clean:
    rm -rf dist/
dist: clean
    python setup.py sdist bdist_wheel
release: dist
    twine upload dist/*

```

* In bash command: `make release` 

A typical folder should look like this:
??? setup.py
??? src
    ??? PACKAGE_NAME
        ??? __init__.py
	??? data
	?   ??? data.pkl
	??? MODULE.py


Access package data

```python

import pkgutil
from pickle import loads

loads(pkgutil.get_data(
    'PACKAGE_NAME',
    'data/numbers.pkl'
))

```

Access package model

```python

import pkg_resources
from joblib import load

load(pkg_resources.resource_filename(
    'PACKAGE_NAME',
    'models/knn.pkl'
))

```


## Project

### Project Templates

Use templates to avoid repetitive tasks (of building a project).
COOKIECUTER project template package.

Usage:

```python

from cookiecutter import main

main.cookiecutter(TEMPLATE_REPO)

```

This will prompt a conversion to initializa a project.

Alternatively, we could suppress the prompt:

```python

from cookiecutter import main

main.cookiecutter(
    'https://github.com/marskar/cookiecutter', ## Alternatively, we can write 'gh:marskar/cookiecutter'
    no_input=True, ## Use defaults
    extra_context={'KEY': 'VALUE'} ## Overide defaults
    
)

## Load local json files as dict

from json import load
from pathllib import path

load(Path(JSON_PATH).open()).values()

## Load remote json

from requests import get

get(JSON_URL).json().values()


# Write a json file
json_path.write_text(json.dumps({
    "project": "Creating Robust Python Workflows",
  	# Convert the project name into snake_case
    "package": "{{ cookiecutter.project.lower().replace(' ', '_') }}",
    # Fill in the default license value
    "license": ["MIT", "BSD", "GPL3"]
}))

pprint(json.loads(json_path.read_text()))

# Obtain keys from the local template's cookiecutter.json
keys = [*json.load(json_path.open())]
vals = "Your name here", "My Amazing Python Project"

# Create a cookiecutter project without prompting for input
main.cookiecutter(template_root.as_posix(), no_input=True,
                  extra_context=dict(zip(keys, vals)))

for path in pathlib.Path.cwd().glob("**"):
    print(path)

```

Sphinx and Read the Docs are friends to establish your project documentation
