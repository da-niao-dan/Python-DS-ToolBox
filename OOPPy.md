# OOP in Python
I recommend this [course](https://learn.datacamp.com/courses/object-oriented-programming-in-python) alongside with official python document,

## Basics of OOP

```python
# Create class DataShell
class DataShell:
    """This is the doc string."""
    family = "fakepd_df" ## this is a class variable
    # Define initialization method
    def __init__(self, filepath):  ## This is a constructor
        self.filepath = filepath ## this is an instance variable
        self.data_as_csv = pd.read_csv(filepath) ## this is an instance variable
    
    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name): ## This is an instance method
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Instantiate DataShell as us_data_shell with argument us_life_expectancy
us_data_shell = DataShell(us_life_expectancy)

# Print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

# Rename your objects column 'code' to 'country_code'
us_data_shell.rename_column('code', 'country_code')

# Again, print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

```

## Best Practices

* Read other's code on GitHub, especially APIs. Pandas, Spark
* PEP-8
* Separation of Concerns

## Inheritance, polymorphism and composition
```python
# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)
```

```python
# Define abstract class DataShell
class DataShell:
    # Class variable family
    family = 'DataShell'
    # Initialization method with arguments, and instance variables
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell      
class CsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data= pd.read_csv(filepath)  ### Yeah, using other component's methods is composition
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell
us_data_shell = CsvDataShell("US", us_life_expectancy)

# Print us_data_shell.stats
print(us_data_shell.stats)

```

