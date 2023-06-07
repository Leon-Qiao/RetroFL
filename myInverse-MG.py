import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

num_nodes = 20480
num_node_epochs = 5000

class DataLoader:
    def __init__(self):
        AS_dataset = pd.read_csv('/user/work/ri22467/20-25-30-35-40.csv', encoding='utf-8')
        self.X = AS_dataset.loc[:,'freq':'l2'].to_numpy()
        self.y = AS_dataset.loc[:,'s11r':'s41i'].to_numpy()
        self.mmX = MinMaxScaler()
        self.X[:,1:] = self.mmX.fit_transform(self.X[:,1:])
        self.X[:,0] = self.X[:,0] / 10
        # self.X, _, self.y, _ = train_test_split(self.X, self.y, test_size=0.75, random_state=0)
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

X_true = np.array([[2.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [2.5, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [3.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02]])
y_true = np.array([[0.054, 0.229, 0.080, -0.711, -0.585, -0.085, -0.262, -0.073], [-0.005, 0.002, -0.458, -0.674, -0.466, 0.328, -0.030, 0.006], [-0.232, -0.110, -0.676, -0.154, -0.118, 0.601, -0.065, -0.248]])
X_true[:,1:] = data_loader.mmX.transform(X_true[:,1:])
X_true[:,0] = X_true[:,0] / 10

def train():
    num_batch = data_loader.num_train // batch_size
    for epoch_index in range(num_epochs):
        for batch in range(num_batch):
            X, y = data_loader.get_batch(batch_size)
            with tf.GradientTape() as tape:
                y_pred = model(X)
                tr_mse = tf.reduce_mean(tf.square(y_pred - y))
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
            y_t_p = model(X_true)
            true_mse = tf.reduce_mean(tf.square(y_t_p - y_true))
            true_rmse = tf.sqrt(true_mse)
            true_mae = tf.reduce_mean(tf.abs(y_t_p - y_true))
            true_r2 = 1 - tf.reduce_sum(tf.square(y_t_p - y_true)) / tf.reduce_sum(tf.square(y_true - tf.cast(tf.reduce_mean(y_true), dtype=tf.float32)))
            print("true mse:{} rmse:{} mae:{} r2:{}".format(true_mse, true_rmse, true_mae, true_r2))

# train()
# tf.saved_model.save(model, './models')

model = tf.saved_model.load('./models')

def obj_func(s_para):
    s_para = tf.square(s_para)
    E11 = s_para[:,0] + s_para[:,1]
    E21 = s_para[:,2] + s_para[:,3]
    E31 = s_para[:,4] + s_para[:,5]
    E41 = s_para[:,6] + s_para[:,7]
    loss = E11 + E41 + tf.abs(E21/(E31+E21) - 0.667)
    return loss

bestLoss = 10
bestStructure = []

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.03)

# mmin = np.min(data_loader.X[: , 1: ], axis=0)
# mmax = np.max(data_loader.X[: , 1: ], axis=0)
# start = np.random.uniform(mmin, mmax, (num_nodes * 2, mmin.shape[0]))

num_gpu = len(gpus)

start = np.random.uniform(0, 1, (num_nodes * num_gpu, 10))
structure = []
for i in range(num_gpu):
    structure.append(tf.Variable(start[i * num_nodes: (i + 1) * num_nodes], dtype=tf.float32))

freqs = [tf.tile([[0.24]], [num_nodes, 1]), tf.tile([[0.25]], [num_nodes, 1]), tf.tile([[0.26]], [num_nodes, 1])]
loss = [0, 0]

for i in range(num_node_epochs):
    for j in range(num_gpu):
        loss[j] = 0
        with tf.device("/gpu:" + str(j)):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(structure[j])
                for k in range(3):
                    y_pred = model(tf.concat([freqs[k], structure[j]], axis=1))
                    loss[j] += obj_func(y_pred)
            grads = tape.gradient(loss[j], structure[j])
            opt.apply_gradients(grads_and_vars=zip([grads], [structure[j]]))
            nega_place = tf.where(structure[j] < 0)
            # structure[j] = tf.Variable(tf.tensor_scatter_nd_update(structure[j], [nega_place], [np.random.uniform(mmin[nega_place[:,1]], mmax[nega_place[:,1]], (nega_place.shape[0]))]))
            structure[j] = tf.Variable(tf.tensor_scatter_nd_update(structure[j], [nega_place], [np.random.uniform(0, 1, (nega_place.shape[0]))]))
    t_min_loss1 = tf.reduce_min(loss[0]).numpy()
    t_min_loss2 = tf.reduce_min(loss[1]).numpy()
    if np.min([t_min_loss1, t_min_loss2]) < bestLoss:
        if t_min_loss1 < t_min_loss2:
            bestLoss = t_min_loss1
            # bestStructure = structure[0][tf.argmin(loss[0])].numpy()
            bestStructure = data_loader.mmX.inverse_transform([structure[0][tf.argmin(loss[0])].numpy()])[0]
            print(tf.argmin(loss[0]).numpy())
        else:
            bestLoss = t_min_loss2
            # bestStructure = structure[1][tf.argmin(loss[1])].numpy()
            bestStructure = data_loader.mmX.inverse_transform([structure[1][tf.argmin(loss[1])].numpy()])[0]
            print(tf.argmin(loss[1]).numpy())
        print(i, bestLoss)
        print(bestStructure)
        print()