import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = tf.saved_model.load('./models')

AS_dataset = pd.read_csv('/user/work/ri22467/20-25-30-35-40.csv', encoding='utf-8')
# dX = AS_dataset.loc[:,'freq':'l2'].to_numpy()
# mmX = MinMaxScaler()
# dX[:,1:] = mmX.fit_transform(dX[:,1:])

X = np.array([[2.7639604,  3.7490022,  2.7388225,  0.5229501,  7.2934303,  7.218549,  2.5859375,  3.546005,  17.558346,  14.7351265]])

# X = mmX.transform(X)

X1 = tf.concat([tf.constant([[2.4]]), X], axis = 1)
X2 = tf.concat([tf.constant([[2.5]]), X], axis = 1)
X3 = tf.concat([tf.constant([[2.6]]), X], axis = 1)

y1 = model(X1)
y2 = model(X2)
y3 = model(X3)

print(y1)
print(y2)
print(y3)

def obj_func(s_para):
    s_para = tf.square(s_para)
    E11 = s_para[:,0] + s_para[:,1]
    E21 = s_para[:,2] + s_para[:,3]
    E31 = s_para[:,4] + s_para[:,5]
    E41 = s_para[:,6] + s_para[:,7]
    loss = E11 + E41 + tf.abs(E21/(E31+E21) - 0.667)
    return loss

print(obj_func(y1))
print(obj_func(y2))
print(obj_func(y3))