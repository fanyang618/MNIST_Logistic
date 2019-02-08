
# coding: utf-8

# In[41]:


from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression

mnist_raw = loadmat('mnist-original.mat')
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

train_image, test_image, train_label, test_label = train_test_split(
    mnist["data"], mnist["target"], test_size=1/7.0, random_state=0)
# ‘lbfgs’ handle multinomial loss
logReg = LogisticRegression(solver = 'lbfgs')
logReg.fit(train_image, train_label)

logReg.predict(test_image)

score = logReg.score(test_image, test_label)
print(score)


