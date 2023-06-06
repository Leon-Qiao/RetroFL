import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# gpus = tf.config.list_physical_devices(device_type = 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

num_nodes = 2048
num_node_epochs = 50000

class DataLoader:
    def __init__(self):
        AS_dataset = pd.read_csv('/user/work/ri22467/20-25-30-35-40.csv', encoding='utf-8')
        self.X = AS_dataset.loc[:,'freq':'l2'].to_numpy()
        self.y = AS_dataset.loc[:,'s11r':'s41i'].to_numpy()
#         self.mmX = MinMaxScaler()
#         self.X[:,1:] = self.mmX.fit_transform(self.X[:,1:])
#         self.X[:,0] = self.X[:,0] / 10
#         self.X, _, self.y, _ = train_test_split(self.X, self.y, test_size=0.75, random_state=0)
        self.X_train, self.X_vali, self.y_train, self.y_vali = train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        self.num_train = self.X_train.shape[0]
    def get_batch(self, batch_size=0, mode='train'):
        if mode == 'train':
            index = np.random.randint(0, self.num_train, batch_size)
            return self.X_train[index], self.y_train[index]
        if mode == 'validate':
            return self.X_vali, self.y_vali
        
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=8,  activation=tf.nn.tanh)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = self.dense5(x)
        return output
    
num_epochs = 150
batch_size = 1024
learning_rate = 0.001


model = MLP()
data_loader = DataLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
X_v, y_v = data_loader.get_batch(mode='validate')
def train():
    num_batch = data_loader.num_train // batch_size
    for epoch_index in range(num_epochs):
        for batch in range(num_batch):
            X, y = data_loader.get_batch(batch_size)
            with tf.GradientTape() as tape:
                y_pred = model(X)
                tr_mse = tf.reduce_mean(tf.square(y_pred - y))
                tr_rmse = tf.sqrt(tr_mse)
                tr_mae = tf.reduce_mean(tf.abs(y_pred - y))
                tr_r2 = 1 - tf.reduce_sum(tf.square(y_pred - y)) / tf.reduce_sum(tf.square(y - tf.cast(tf.reduce_mean(y), dtype=tf.float32)))
            grads = tape.gradient(tr_mse, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if epoch_index in [0, 25, 50, 75, 100, 125, 149]:
            tr_rmse = tf.sqrt(tr_mse)
            tr_mae = tf.reduce_mean(tf.abs(y_pred - y))
            tr_r2 = 1 - tf.reduce_sum(tf.square(y_pred - y)) / tf.reduce_sum(tf.square(y - tf.cast(tf.reduce_mean(y), dtype=tf.float32)))
            print("epoch:{}".format(epoch_index))
            print("train mse:{} rmse:{} mae:{} r2:{}".format(tr_mse, tr_rmse, tr_mae, tr_r2))
            y_v_p = model(X_v)
            va_mse = tf.reduce_mean(tf.square(y_v_p - y_v))
            va_rmse = tf.sqrt(va_mse)
            va_mae = tf.reduce_mean(tf.abs(y_v_p - y_v))
            va_r2 = 1 - tf.reduce_sum(tf.square(y_v_p - y_v)) / tf.reduce_sum(tf.square(y_v - tf.cast(tf.reduce_mean(y_v), dtype=tf.float32)))
            print("vali mse:{} rmse:{} mae:{} r2:{}".format(va_mse, va_rmse, va_mae, va_r2))
            X_true = np.array([[2.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [2.5, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [3.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02]])
            y_true = np.array([[0.054, 0.229, 0.080, -0.711, -0.585, -0.085, -0.262, -0.073], [-0.005, 0.002, -0.458, -0.674, -0.466, 0.328, -0.030, 0.006], [-0.232, -0.110, -0.676, -0.154, -0.118, 0.601, -0.065, -0.248]])
#           X_true[:,1:] = self.data_loader.mmX.transform(X_true[:,1:])
            y_t_p = model(X_true)
            true_mse = tf.reduce_mean(tf.square(y_t_p - y_true))
            true_rmse = tf.sqrt(true_mse)
            true_mae = tf.reduce_mean(tf.abs(y_t_p - y_true))
            true_r2 = 1 - tf.reduce_sum(tf.square(y_t_p - y_true)) / tf.reduce_sum(tf.square(y_true - tf.cast(tf.reduce_mean(y_true), dtype=tf.float32)))
            print("true mse:{} rmse:{} mae:{} r2:{}".format(true_mse, true_rmse, true_mae, true_r2))

train()
tf.saved_model.save(model, './models')

# model = tf.saved_model.load('./models')

def obj_func(s_para):
    E11 = s_para[:,0]**2 + s_para[:,1]**2
    E21 = s_para[:,2]**2 + s_para[:,3]**2
    E31 = s_para[:,4]**2 + s_para[:,5]**2
    E41 = s_para[:,6]**2 + s_para[:,7]**2
    loss = E11 + E41 + tf.abs(E21/(E31+E21) - 0.667)
    return loss

bestLoss = 10
bestStructure = []

opt = tf.keras.optimizers.Adam(learning_rate=0.003)
rindex = np.random.randint(0, data_loader.X.shape[0], num_nodes)
start = data_loader.X[rindex, 1:]
structure = tf.Variable(start, dtype=tf.float32)
for i in range(num_node_epochs):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(structure)
        X1 = tf.concat([tf.tile([[2.4]], [num_nodes, 1]), structure], axis=1)
        X2 = tf.concat([tf.tile([[2.5]], [num_nodes, 1]), structure], axis=1)
        X3 = tf.concat([tf.tile([[2.6]], [num_nodes, 1]), structure], axis=1)
        y1_pred = model(X1)
        y2_pred = model(X2)
        y3_pred = model(X3)
        loss = obj_func(y1_pred) + obj_func(y2_pred) + obj_func(y3_pred)
        t_min_loss = tf.reduce_min(loss).numpy()
        if t_min_loss < bestLoss:
            bestLoss = t_min_loss
#             bestStructure = A.data_loader.mmX.inverse_transform([structure.numpy()[0]])[0]
            bestStructure = structure[tf.argmin(loss)]
            print(tf.argmin(loss).numpy(), i, bestLoss)
            print(bestStructure)
            print()
    grads = tape.gradient(loss, structure)
    opt.apply_gradients(grads_and_vars=zip([grads], [structure]))
    nega_place = tf.where(structure < 0)
    structure = tf.Variable(tf.tensor_scatter_nd_update(structure, [nega_place], [data_loader.X[np.random.randint(0, data_loader.X.shape[0], nega_place.shape[0]), nega_place[:,1] + 1]]))