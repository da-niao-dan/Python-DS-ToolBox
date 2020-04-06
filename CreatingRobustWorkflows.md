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

## Documentation and Types
