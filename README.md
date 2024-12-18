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

---

### [Lab Task 3](Ai-Lab-Work-Copy-Paste/Lab%20Task%203/)
In **Lab Task 3**, we worked with the **data.csv** file for data analysis using Pandas. The dataset includes columns like 'job', 'education', 'contact', and others, with 126,030 rows and 8 columns.

Key tasks performed:

1. **Read and Inspect Data**:
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   df.head()
   df.shape
   df.info()
   ```

2. **Check Missing Values**:
   ```python
   df.isnull()
   df.isnull().sum()
   ```

3. **Change Data Types**:
   ```python
   for col in df.select_dtypes(include=['object']).columns:
       df[col] = df[col].astype('category')
   df.dtypes
   ```

4. **Descriptive Statistics**:
   ```python
   df.describe().T
   ```

5. **Group and Aggregate Data**:
   ```python
   df.groupby('BASEL_BBQ_weather').mean()
   ```

6. **Check Presence of a Value**:
   ```python
   is_present = True in df['BASEL_BBQ_weather'].values
   is_present
   ```

7. **Access Specific Data**:
   ```python
   value_at_index = df.iloc[5]
   value_at_index
   ```

---

## Dataset Details
The dataset used in **Lab Task 3** contains the following:

- **Shape**: 126,030 rows and 8 columns.
- **Columns**:
  - **Categorical**: 'job', 'education', 'contact', 'month', 'day_of_week', 'y'
  - **Numerical**: 'age', 'duration'
- **No Missing Values** were found.
- **Key Statistics**: Numerical columns include measures such as mean, min, max, and quartiles, while categorical columns include frequencies and unique value counts.

For more details, see the [Lab Task 3 Directory](Ai-Lab-Work-Copy-Paste/Lab%20Task%203/).

---

## Additional Lab Directories

The repository also contains other lab tasks, each focused on specific concepts. You can explore them below:

- [Lab Task 4](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%204/)
- [Lab Task 6](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%206/)
- [Lab Task 8](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%208/)
- [Lab Task 9](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%209/)

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
