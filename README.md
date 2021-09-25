# Deep Learning Teaching Examples (Python - Keras)

Deep learning examples used for teaching within the School of Computing at [Newcastle University](http://www.ncl.ac.uk) (UK) by [Dr. Amir Atapour-Abarghouei](http://www.atapour.co.uk/).

The material is presented as part of the "Deep Learning" lecture series at Newcastle University (CSC8637).

All material here has been tested with [Keras](https://keras.io/) 2.4 and Python 3.6.

---

### Background:

All the code examples are available in the form of [a notebook](https://jupyter.org/). A simple Python file is also available for every example. The notebooks and the python files are equivalent in every way. Both formats are provided for convenience.

It is recommended that all code examples are run using a GPU. Training is done using publicly available datasets such as [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

---

### Running the Code:

- You may download each file as needed.
- You can also download the entire repository as follows:

```
git clone https://github.com/atapour/keras-dl-examples
cd keras-dl-examples
cd <sub directory of one of the examples>
```
In each sub-directory, you can find:

+ .py file - python code for the examples
+ .ipynb file - a python notebook with detailed descriptions for running the examples

- You can simply run a Python file by navigating to its corresponding sub-directory and running:

```
python <example file name>.py
```

If you do not have direct access to a GPU, the examples can be run via [Google Colab](https://colab.research.google.com).
### Running the Code in Google Colab:

 - Navigate to - [https://colab.research.google.com](https://colab.research.google.com)
 - Sign in with a Google account.

#### Using Google Colab Directly from Github:
- Select File -> Upload Notebook... -> Github
- Paste a URL of an `.ipynb` file provided in this repository.

#### Uploading the Notebook from the Local Copy:

 - Select File -> Upload Notebook...
 - Drag and drop or browse to select the notebook you wish to use from this repository cloned to your local machine.


 ### Important Notes:

 - As the examples are best run using a GPU, make sure you enable the use of a GPU in [Google Colab](https://colab.research.google.com).

 - Select Runtime -> Change runtime type -> GPU

---

All code is provided _"as is"_ to aid learning and understanding of topics within the "Deep Learning" course.

---

Please raise an issue in this repository if you find any bugs.
It would even be better if you submitted a pull request with a fix.
