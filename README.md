# Artificial Intelligence Lab Tasks

This repository contains the lab tasks completed during Artificial Intelligence coursework. Each lab is organized into separate directories and contains tasks aimed at building foundational skills in AI and Python programming. Below is an overview of the tasks completed in each lab.

---

## Lab 1: Importing Libraries

In this lab, we learned how to import and utilize essential Python libraries that are commonly used in Artificial Intelligence and Data Science projects. The libraries covered include:

### Libraries Learned:

1. **Numpy**: A library for numerical computations, particularly useful for working with arrays and matrices. It provides a high-performance multidimensional array object and tools for working with these arrays.
    - Example: `import numpy as np`

2. **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow, designed for fast experimentation.
    - Example: `from keras.models import Sequential`

3. **Pandas**: A powerful data analysis and manipulation library, providing data structures like DataFrames for managing structured data efficiently.
    - Example: `import pandas as pd`

4. **Sklearn (Scikit-learn)**: A machine learning library that includes tools for classification, regression, clustering, and dimensionality reduction.
    - Example: `from sklearn.model_selection import train_test_split`

### Lab Directory
[Lab 1: Importing Libraries](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%201/)

---

## Lab 2: Python Basics

In this lab, we explored Python basics, including variable input, string manipulations, and simple functions. Below are the details of what was covered:

### Tasks and Functions:

#### 1. **Variable Input and Arithmetic Operations**

```python
user = int(input())  # Input: 5
print(user)          # Output: 5

# Adding two integers:
a = int(input('a = '))  # Input: 3
b = int(input('b = '))  # Input: 4
print(a + b)            # Output: 7
```

#### 2. **String Manipulations**

```python
user = 'My Name is HAMNA'
user.lower()          # Output: 'my name is hamna'
user.upper()          # Output: 'MY NAME IS HAMNA'
user.swapcase()       # Output: 'mY nAME IS hamna'
user.title()          # Output: 'My Name Is Hamna'
```

**Details of Functions:**
- `lower()`: Converts all characters in the string to lowercase.
    - Input: "My Name is HAMNA"
    - Output: "my name is hamna"
- `upper()`: Converts all characters in the string to uppercase.
    - Input: "My Name is HAMNA"
    - Output: "MY NAME IS HAMNA"
- `swapcase()`: Swaps the case of each character in the string.
    - Input: "My Name is HAMNA"
    - Output: "mY nAME IS hamna"
- `title()`: Converts the first character of each word to uppercase and the rest to lowercase.
    - Input: "My name is HAMNA"
    - Output: "My Name Is Hamna"

#### 3. **String Length and Count**

```python
a = 'My name is my Identity'
print(len(a))        # Output: 23
print(a.count('my')) # Output: 2
```

**Details of Functions:**
- `len()`: Returns the total number of characters in the string, including spaces.
    - Input: "My name is my Identity"
    - Output: `23`
- `count()`: Counts the occurrences of a substring in the string.
    - Input: `a.count('my')`
    - Output: `2`

### Lab Directory
[Lab 2: Python Basics](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%202/)

---

# Lab Task 3: Pandas Data Manipulation
This lab focuses on performing data analysis and manipulation using the Pandas library in Python. We explored various functions and methods to understand and preprocess the dataset provided in the same directory.

## Dataset Description
The dataset, `data.csv`, contains information about weather conditions and related metrics. It includes the following columns:

1. **BASEL_BBQ_weather**: Weather conditions for BBQ in Basel.
2. **Temperature**: Numerical values indicating temperature.
3. **Humidity**: Numerical values indicating humidity levels.
4. **Wind_Speed**: Numerical values indicating wind speed.

### Preview of the Dataset
A quick look at the first five rows of the dataset can be achieved using the `head()` function:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('data.csv')
print(df.head())
```
Output:
| BASEL_BBQ_weather | Temperature | Humidity | Wind_Speed |
|--------------------|-------------|----------|------------|
| Sunny             | 25          | 50       | 10         |
| Cloudy            | 22          | 60       | 12         |
| Rainy             | 18          | 80       | 5          |
| Sunny             | 30          | 40       | 8          |
| Cloudy            | 20          | 70       | 15         |

## Code Walkthrough and Explanations

### 1. Importing Libraries
```python
import pandas as pd
```
We imported Pandas for efficient data handling and manipulation.

### 2. Reading the Dataset
```python
df = pd.read_csv('data.csv')
```
The dataset `data.csv` was loaded into a Pandas DataFrame named `df`.

### 3. Dataset Overview
- **`df.head()`**: Displays the first 5 rows of the dataset.
    - Input: None
    - Output: First 5 rows of the DataFrame.
- **`df.shape`**: Returns the number of rows and columns in the dataset.
    - Input: None
    - Output: (Number of Rows, Number of Columns)
- **`df.info()`**: Provides a concise summary of the dataset, including non-null counts and data types for each column.
    - Input: None
    - Output: DataFrame summary.

### 4. Handling Missing Values
- **`df.isnull()`**: Checks for missing values in the dataset. Returns a DataFrame of the same shape with `True` for missing and `False` for non-missing values.
    - Input: None
    - Output: DataFrame with boolean values indicating missing data.
- **`df.isnull().sum()`**: Provides the total count of missing values for each column.
    - Input: None
    - Output: Series with column names as index and missing value counts as values.

### 5. Converting Object Columns to Categorical
```python
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
```
This code snippet converts all object-type columns to categorical type for efficient storage and faster processing.
- **`df.select_dtypes(include=['object'])`**: Selects columns with data type `object`.
- **`astype('category')`**: Changes the data type to `category`.

### 6. Data Types
```python
print(df.dtypes)
```
- This prints the data types of all columns in the DataFrame after the conversions.

### 7. Statistical Summary
```python
print(df.describe().T)
```
- **`df.describe()`**: Generates summary statistics for numerical columns (e.g., count, mean, standard deviation, min, max).
- **`.T`**: Transposes the output for a clearer view.

### 8. Grouping Data
```python
print(df.groupby('BASEL_BBQ_weather').mean())
```
- Groups the dataset by `BASEL_BBQ_weather` and calculates the mean of each numeric column for each group.

### 9. Checking Value Presence
```python
is_present = True in df['BASEL_BBQ_weather'].values
print(is_present)
```
- **`in` operator**: Checks if `True` is present in the column `BASEL_BBQ_weather`.
    - Output: `True` or `False`.

### 10. Accessing Specific Rows
```python
value_at_index = df.iloc[5]
print(value_at_index)
```
- **`df.iloc[5]`**: Retrieves the row at index 5.
    - Output: Data of the specific row.

---
## Lab Directory
[Lab Task 3: Pandas Data Manipulation](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%203/)

---

## Repository Structure

The repository is organized as follows:
```
Ai-Lab-Work-Copy-Paste/
|
|-- Lab Task 1/
|-- Lab Task 2/
|-- Lab Task 3/
|-- Lab Task 4/
|-- Lab Task 6/
|-- Lab Task 8/
|-- Lab Task 9/
```

### How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste.git
   ```
2. Navigate to the desired lab directory to explore the tasks and solutions.

---

### Contributions
Feel free to contribute by improving the code or adding new features. Submit a pull request with your changes.

---

### License
This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
