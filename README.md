# CS412

# Getting Start
## Prerequisites
python 3.6.0 or higher version
library: numpy, pandas, collections, sklearn, matplotlib, pickle

## Installing
Open command line
```
git clone https://github.com/JYLEvageline/CS412.git
```

# Running the tests
If you want to run the whole project, please type the following in command line:
```
python __init__.py
```

If you want to know how the data processing works, please run:
```python
from read import data_process
data_process('responses.csv')
```
It would clean the data and create the training set, test set and validation set.

If you want to know how to train my model, please type:
```python
from train import my_model
my_model()
```
It would train the model, including feature selection and hyperparameter selection. Then it would save the model just trained.

If you want to know how to train baseline, please type:
```python
from baseline import baseline_models
baseline_models()
```
It would train the baseline models, including hyperparameter selection. Then it would save the models just trained.

Finally, if you want test my model, please type:
```python
from __init__ import testmy
testmy()
```
