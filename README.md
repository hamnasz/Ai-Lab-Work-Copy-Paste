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
# Lab Task 4: Linear Regression with Data Visualization

This lab focuses on implementing a simple linear regression model using Python libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn. Additionally, the task involves visualizing the regression results using a scatter plot and line graph.

---

## Objective
To develop a foundational understanding of linear regression by:
1. Preparing synthetic and real-world datasets.
2. Splitting data into training and testing sets.
3. Training a linear regression model.
4. Making predictions and visualizing the results.

---

## Steps Performed

### 1. **Importing Required Libraries**
The following libraries were used:
- **NumPy**: For generating random numbers and numerical computations.
- **Pandas**: For loading and analyzing the dataset.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For splitting data into train/test sets and implementing linear regression.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

---

### 2. **Dataset Overview**

#### Synthetic Dataset:
- Features (‘x’): Generated using NumPy random functions.
- Target (‘y’): Generated using a linear equation with noise.

```python
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)
```

#### Real-World Dataset:
- The provided dataset, `data.csv`, contains the following columns:
  - **Date**: Date of activity.
  - **Usage**: Time spent using the app.
  - **Notifications**: Number of notifications received.
  - **Times Opened**: Frequency of app openings.
  - **App**: Name of the application.

```python
df = pd.read_csv('data.csv')
print(df.head())
```

#### Example Data:
| Date       | Usage | Notifications | Times Opened | App       |
|------------|-------|---------------|--------------|-----------|
| 08/26/2022 | 38    | 70            | 49           | Instagram |
| 08/27/2022 | 39    | 43            | 48           | Instagram |
| 08/28/2022 | 64    | 231           | 55           | Instagram |
| 08/29/2022 | 14    | 35            | 23           | Instagram |
| 08/30/2022 | 3     | 19            | 5            | Instagram |

---

### 3. **Data Splitting**
The synthetic dataset was split into training and testing sets using an 80-20 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

---

### 4. **Model Training**
A linear regression model was trained using Scikit-learn's `LinearRegression` class.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

### 5. **Prediction**
The trained model was used to predict target values for the test dataset.

```python
y_pred = model.predict(X_test)
```

---

### 6. **Visualization**
A scatter plot and regression line were plotted to visualize the model's predictions against actual test data.

```python
plt.scatter(X_test, y_test, color='b', label='Actual Data')
plt.plot(X_test, y_pred, color='r', label='Regression Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
```

#### Visualization Output:
The graph displays a clear linear relationship between the variables, with the regression line closely following the trend of the actual data points.

---

## Key Observations
- The linear regression model effectively learned the relationship between the input and output variables.
- The scatter plot and regression line visualization help assess the model's accuracy visually.
- The dataset `data.csv` provides meaningful information on app usage that can be further explored using similar regression techniques.

---

## Lab Directory
[Lab Task 4: Linear Regression](https://github.com/hamnasz/Ai-Lab-Work-Copy-Paste/tree/main/Lab%20Task%204/)

---

# Lab Task 6:

## File 1: **Crop Image**
### Code:
```python
import cv2
image = cv2.imread("girl.jfif")
cv2.imshow("Original", image)
face = image[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)
```
### Functions:
1. **`cv2.imread("girl.jfif")`**:
   - Reads the input image file `"girl.jfif"` into a NumPy array.
   - Mode: Default (color image).

2. **`cv2.imshow("Original", image)`**:
   - Displays the original image in a new window titled `"Original"`.

3. **`image[85:250, 85:220]`**:
   - Crops the specified rectangular region from the image (`[y1:y2, x1:x2]`).
   - Extracts the "Face" region.

4. **`cv2.imshow("Face", face)`**:
   - Displays the cropped face region.

5. **`image[90:450, 0:290]`**:
   - Crops another region from the image corresponding to the "Body."

6. **`cv2.imshow("Body", body)`**:
   - Displays the cropped body region.

7. **`cv2.waitKey(0)`**:
   - Waits indefinitely for a key press to close the image window.

---

## File 2: **Image Processing**
### Code:
```python
import cv2
import numpy as np
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
eroded_image = cv2.erode(image, kernel, iterations=1)
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### Functions:
1. **`cv2.IMREAD_GRAYSCALE`**:
   - Loads the image (`image.jpg`) in grayscale mode.

2. **`np.ones((kernel_size, kernel_size), np.uint8)`**:
   - Creates a square kernel (5x5) filled with ones, used for morphological operations.

3. **`cv2.erode(image, kernel, iterations=1)`**:
   - Erodes the input image to reduce noise or thin the boundaries of objects.
   - One iteration is performed.

4. **`cv2.imshow('Original Image', image)`**:
   - Displays the original grayscale image.

5. **`cv2.imshow('Eroded Image', eroded_image)`**:
   - Displays the eroded image.

6. **`cv2.destroyAllWindows()`**:
   - Closes all open image windows.

---

## File 3: **Load and Display**
### Code:
```python
import argparse
import cv2
import sys
sys.argv = ['']
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")

args = vars(ap.parse_args(["--image", "anime.jfif"]))
image = cv2.imread(args["image"])
print("width: {w} pixels".format(w=image.shape[1]))
print("height: {h}  pixels".format(h=image.shape[0]))
print("channels: {c}".format(c=image.shape[2]))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imwrite("newimage.jpg", image)
```
### Functions:
1. **`argparse.ArgumentParser()`**:
   - Used for parsing command-line arguments. Adds an argument for the image path.

2. **`cv2.imread(args["image"])`**:
   - Reads the image specified in the command-line argument (`anime.jfif`).

3. **Image Properties**:
   - `image.shape[1]`: Width of the image.
   - `image.shape[0]`: Height of the image.
   - `image.shape[2]`: Number of color channels (e.g., RGB has 3 channels).

4. **`cv2.imshow("Image", image)`**:
   - Displays the loaded image.

5. **`cv2.imwrite("newimage.jpg", image)`**:
   - Saves the displayed image to a new file (`newimage.jpg`).

---

## File 4: **Morphological Operations**
### Code:
```python
import argparse
import cv2
args = {"image": "rene.jfif"}
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
...
```
### Functions:
1. **`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`**:
   - Converts the image to grayscale.

2. **Erosion (`cv2.erode`)**:
   - Shrinks bright regions. Iterative erosion is performed three times, and each result is displayed.

3. **Dilation (`cv2.dilate`)**:
   - Expands bright regions. Iterative dilation is performed three times.

4. **Opening (`cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)`)**:
   - Removes small objects from the foreground (erosion followed by dilation).

5. **Closing (`cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)`)**:
   - Fills small holes in the foreground (dilation followed by erosion).

6. **Gradient (`cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)`)**:
   - Highlights the edges of objects by computing the difference between dilation and erosion.

7. **Kernel Sizes**:
   - Morphological operations are applied with varying kernel sizes (`(3x3)`, `(5x5)`, `(7x7)`).

---
### Lab 8 Explanation: Building an Artificial Neural Network (ANN)

The provided code demonstrates how to preprocess data and build a binary classification ANN using **TensorFlow** and **Keras**.

---

### Detailed Steps and Functions

#### **1. Import Libraries**
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
```
- **`tensorflow`**: Framework for building and training deep learning models.
- **`numpy`**: Used for numerical operations on arrays.
- **`pandas`**: Provides tools for data manipulation.
- **`sklearn.metrics.accuracy_score`**: Evaluates the accuracy of predictions.

---

#### **2. Load and Prepare the Dataset**
```python
data = pd.read_csv("data.csv")
X = data.iloc[:,3:-1].values
Y = data.iloc[:,-1].values
```
- **`pd.read_csv("data.csv")`**: Reads the dataset from a CSV file.
- **`data.iloc[:,3:-1]`**:
  - Selects feature columns starting from the 4th column (index 3) to the second-last column.
  - **`X`**: Input features.
- **`data.iloc[:,-1]`**:
  - Selects the last column as the target variable.
  - **`Y`**: Target labels.

---

#### **3. Encode Categorical Data**
```python
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))
```
- **`LabelEncoder`**:
  - Encodes categorical features into numerical values (e.g., "Male" to 0, "Female" to 1).
  - Applied to the third column (`X[:,2]`) of the dataset.

---

#### **4. One-Hot Encoding**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
```
- **`ColumnTransformer`**:
  - Applies **OneHotEncoder** to the second column (`[1]`) of the dataset.
  - Converts categorical variables into a binary matrix.
  - Example: "Red", "Green", "Blue" → [1, 0, 0], [0, 1, 0], [0, 0, 1].
- **`remainder="passthrough"`**:
  - Keeps the rest of the columns unchanged.

---

#### **5. Split the Dataset**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
- **`train_test_split`**:
  - Splits the dataset into training and testing sets.
  - **`test_size=0.2`**: 20% of the data is used for testing.
  - **`random_state=0`**: Ensures reproducibility.

---

#### **6. Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
- **`StandardScaler`**:
  - Normalizes features by removing the mean and scaling to unit variance.
  - Ensures all features contribute equally to the model.

---

#### **7. Build the ANN**
```python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```
- **`tf.keras.models.Sequential`**:
  - Initializes a sequential model, where layers are added one after another.

- **First Layer**:
  - **`Dense(units=6)`**: Dense (fully connected) layer with 6 neurons.
  - **`activation="relu"`**: Rectified Linear Unit activation function.

- **Output Layer**:
  - **`Dense(units=1)`**: Output layer with 1 neuron.
  - **`activation="sigmoid"`**:
    - Outputs a probability between 0 and 1 (for binary classification).

---

#### **8. Compile the Model**
```python
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
```
- **`optimizer="adam"`**:
  - Adaptive Moment Estimation (Adam) optimization algorithm.
  - Efficient for large datasets and non-stationary objectives.

- **`loss="binary_crossentropy"`**:
  - Loss function for binary classification tasks.

- **`metrics=['accuracy']`**:
  - Tracks accuracy during training and evaluation.

---

#### **9. Train the Model**
```python
ann.fit(X_train, Y_train, batch_size=32, epochs=100)
```
- **`X_train`**, **`Y_train`**: Training data and labels.
- **`batch_size=32`**:
  - Number of samples processed before updating model weights.
- **`epochs=100`**:
  - Number of complete passes through the training dataset.

---

### Overview of Process
1. Data preprocessing (encoding, scaling, splitting).
2. Building a simple ANN architecture.
3. Compiling the model with appropriate loss and optimizer.
4. Training the model over 100 epochs.

---
### Lab 9 Explanation: Unzipping and Processing Images

This program demonstrates how to:
1. Extract a ZIP file containing images.
2. Process images by displaying them in **original**, **grayscale**, and **simulated NIR (Near Infrared)** formats.

---

### Step-by-Step Explanation

#### **1. Import Necessary Libraries**
```python
import os
import zipfile
from IPython.display import display, Image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
```
- **`os`**: Handles file and directory paths.
- **`zipfile`**: Extracts files from a ZIP archive.
- **`PIL (Pillow)`**: For image processing (e.g., opening, editing, and converting images).
- **`matplotlib.pyplot`**: Visualizes images in various formats.

---

#### **2. Define Paths**
```python
directory_to_extract_to = 'extracted_images'
```
- **`directory_to_extract_to`**: Specifies the folder where ZIP files will be extracted.  
  Example Path: `'extracted_images/'`

---

#### **3. Extract the ZIP File**
```python
with zipfile.ZipFile('Image.zip', 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
```
- **`zipfile.ZipFile('Image.zip', 'r')`**: Opens the ZIP file named `Image.zip` in read mode.
- **`.extractall(directory_to_extract_to)`**: Extracts all files into the folder `extracted_images`.

---

#### **4. Create an Output Folder (If Not Exists)**
```python
extraction_folder = 'image'
if not os.path.exists(extraction_folder):
    os.makedirs(extraction_folder)
```
- **`os.makedirs()`**: Creates the folder `'image'` if it doesn't already exist.
  Example Path: `'image/'`

---

#### **5. Define the Unzipping Function**
```python
def unzip_folder(zip_file_path, extraction_folder):
    # Code to unzip the folder
    pass
```
This placeholder function can be implemented to generalize unzipping functionality.

---

#### **6. List and Display Extracted Images**
```python
extracted_folder = 'extracted_images'

for file_name in os.listdir(extracted_folder):
    if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        image_path = os.path.join(extracted_folder, file_name)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
```
- **`os.listdir()`**: Lists all files in the `extracted_folder`.
- **`os.path.join()`**: Combines folder path with filenames to create complete file paths.
- **`Image.open(image_path)`**: Opens each image file.

Example Files:
- `extracted_images/anime.jfif`
- `extracted_images/girl.jfif`
- `extracted_images/image.jpg`

---

#### **7. Unzip and Process Images**
```python
def unzip_and_process_images(zip_file_path, extract_to_folder):
    if not os.path.exists(zip_file_path):
        print(f"Error: The file {zip_file_path} does not exist.")
        return

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

    image_files = [
        os.path.join(extract_to_folder, file)
        for file in os.listdir(extract_to_folder)
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))
    ]

    if not image_files:
        print("No images found in the extracted folder.")
        return

    for image_file in image_files:
        process_image(image_file)
```
- Checks if the ZIP file exists and extracts it.
- **`image_files`**: Filters and lists all image files with specific extensions.
- Calls **`process_image(file_path)`** for each image file.

---

#### **8. Process Individual Images**
```python
def process_image(file_path):
    try:
        img = Image.open(file_path)
        grayscale_img = img.convert('L')  # Converts to grayscale
        nir_img = img.split()[0]  # Simulates NIR using the red channel

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(grayscale_img, cmap='gray')
        plt.axis('off')
        plt.title('Grayscale Image')

        plt.subplot(1, 3, 3)
        plt.imshow(nir_img, cmap='gray')
        plt.axis('off')
        plt.title('Simulated NIR Image')

        plt.show()
    except Exception as e:
        print(f"Error processing the image {file_path}: {e}")
```
- **`grayscale_img`**: Converts the original image to grayscale.
- **`nir_img`**: Uses the red channel of the image as a simulated NIR image.
- **`plt.figure(figsize=(15, 5))`**: Creates a plot for displaying the images.
- Subplots:
  - Original Image
  - Grayscale Image
  - Simulated NIR Image

Example Paths:
- `'extracted_images/anime.jfif'`
- `'extracted_images/girl.jfif'`
- `'extracted_images/image.jpg'`

---

#### **9. Execute the Script**
```python
zip_file_path = 'images.zip'
extract_to_folder = 'extracted_images'

if not os.path.exists(extract_to_folder):
    os.makedirs(extract_to_folder)

unzip_and_process_images(zip_file_path, extract_to_folder)
```
- **`zip_file_path`**: Path to the ZIP file (`images.zip`).
- **`extract_to_folder`**: Folder where files will be extracted (`extracted_images`).
- Calls **`unzip_and_process_images()`** to execute the process.

---

### Example Output Paths:
1. **ZIP File**: `'images.zip'`
2. **Extracted Folder**: `'extracted_images/'`
3. **Image Files**:
   - `'extracted_images/anime.jfif'`
   - `'extracted_images/girl.jfif'`
   - `'extracted_images/image.jpg'`

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
