import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

num_nodes = 2048
num_node_epochs = 50000

class DataLoader:
    def __init__(self):
        AS_dataset = pd.read_csv('/user/work/ri22467/20-25-30-35-40.csv', encoding='utf-8')
        self.X = AS_dataset.loc[:,'freq':'l2'].to_numpy()
        self.y = AS_dataset.loc[:,'s11r':'s41i'].to_numpy()
        # self.mmX = MinMaxScaler()
        # self.X[:,1:] = self.mmX.fit_transform(self.X[:,1:])
        # self.X[:,0] = self.X[:,0] / 10
        # self.X, _, self.y, _ = train_test_split(self.X, self.y, test_size=0.75, random_state=0)
        self.X_train, self.X_vali, self.y_train, self.y_vali = train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        self.X_train, self.y_train = self.X, self.y
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
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=8,  activation=tf.nn.tanh)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = self.dense5(x)
        return output
    
num_epochs = 200
batch_size = 1024
learning_rate = 0.001

model = MLP()
data_loader = DataLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
X_v, y_v = data_loader.get_batch(mode='validate')

X_true = np.array([[2.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [2.5, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02], [3.0, 2.003, 2.615, 1.335, 1.5, 3.177, 2.684, 1.034, 1.943, 20.81, 17.02]])
y_true = np.array([[0.054, 0.229, 0.080, -0.711, -0.585, -0.085, -0.262, -0.073], [-0.005, 0.002, -0.458, -0.674, -0.466, 0.328, -0.030, 0.006], [-0.232, -0.110, -0.676, -0.154, -0.118, 0.601, -0.065, -0.248]])
# X_true[:,1:] = data_loader.mmX.transform(X_true[:,1:])
# X_true[:,0] = X_true[:,0] / 10

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
        if epoch_index % 10 == 0 or epoch_index == num_epochs - 1:
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

train()
tf.saved_model.save(model, './models')

# model = tf.saved_model.load('./models')

"""
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

mmin = np.min(data_loader.X[: , 1: ], axis=0)
mmax = np.max(data_loader.X[: , 1: ], axis=0)

num_gpu = len(gpus)

structure = []
for i in range(num_gpu):
    # structure.append(tf.Variable(np.random.uniform(0, 1, (num_nodes, 10)), dtype=tf.float32))
    structure.append(tf.Variable(np.random.uniform(mmin, mmax, (num_nodes, 10)), dtype=tf.float32))

freq1 = tf.ones([num_nodes, 1]) * 2.4
freq2 = tf.ones([num_nodes, 1]) * 2.5
freq3 = tf.ones([num_nodes, 1]) * 2.6

minLoss = [0, 0]
minIndex = [0, 0]
minS = [0, 0]

def check(structure):
    inva_place1 = tf.where(tf.logical_or(structure[:,:8] < 0.1, structure[:,:8] > 10.1))
    structure = tf.tensor_scatter_nd_update(structure, [inva_place1], [np.random.uniform(mmin[inva_place1[:,1]], mmax[inva_place1[:,1]], (inva_place1.shape[0]))])
    
    inva_place2 = tf.where(tf.logical_or(structure[:,8:] < 1, structure[:,8:] > 101)) + [0, 8]
    structure = tf.tensor_scatter_nd_update(structure, [inva_place2], [np.random.uniform(mmin[inva_place2[:,1]], mmax[inva_place2[:,1]], (inva_place2.shape[0]))])
    
    inva_place3 = tf.where(structure[:,1] < structure[:,7]) # W2 < W8
    a = tf.concat([inva_place3, tf.ones([inva_place3.shape[0], 1], dtype=tf.int64)], axis=1)
    b = tf.concat([inva_place3, tf.ones([inva_place3.shape[0], 1], dtype=tf.int64) * 7], axis=1)
    ori = tf.concat([a, b], axis=0)
    cht = tf.concat([b, a], axis=0)
    structure = tf.tensor_scatter_nd_update(structure, [ori], [tf.gather_nd(structure, cht)])
    
    inva_place4 = tf.where(structure[:,1] < structure[:,0]) # W2 < W1
    a = tf.concat([inva_place4, tf.ones([inva_place4.shape[0], 1], dtype=tf.int64)], axis=1)
    b = tf.concat([inva_place4, tf.zeros([inva_place4.shape[0], 1], dtype=tf.int64)], axis=1)
    ori = tf.concat([a, b], axis=0)
    cht = tf.concat([b, a], axis=0)
    structure = tf.tensor_scatter_nd_update(structure, [ori], [tf.gather_nd(structure, cht)])
    
    inva_place5 = tf.where(structure[:,4] < structure[:,3]) # W5 < W4
    a = tf.concat([inva_place5, tf.ones([inva_place5.shape[0], 1], dtype=tf.int64) * 4], axis=1)
    b = tf.concat([inva_place5, tf.ones([inva_place5.shape[0], 1], dtype=tf.int64) * 3], axis=1)
    ori = tf.concat([a, b], axis=0)
    cht = tf.concat([b, a], axis=0)
    structure = tf.tensor_scatter_nd_update(structure, [ori], [tf.gather_nd(structure, cht)])
    
    inva_place6 = tf.where(structure[:,4] < structure[:,5]) # W5 < W6
    a = tf.concat([inva_place6, tf.ones([inva_place6.shape[0], 1], dtype=tf.int64) * 4], axis=1)
    b = tf.concat([inva_place6, tf.ones([inva_place6.shape[0], 1], dtype=tf.int64) * 5], axis=1)
    ori = tf.concat([a, b], axis=0)
    cht = tf.concat([b, a], axis=0)
    structure = tf.tensor_scatter_nd_update(structure, [ori], [tf.gather_nd(structure, cht)])
    
    return tf.Variable(structure)
    
    # structure[j] = tf.Variable(tf.tensor_scatter_nd_update(structure[j], [nega_place], [np.random.uniform(0, 1, (nega_place.shape[0]))]))

for i in range(num_node_epochs):
    for j in range(num_gpu):
        with tf.device("/gpu:" + str(j)):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(structure[j])
                y_pred1 = model(tf.concat([freq1, structure[j]], axis=1))
                y_pred2 = model(tf.concat([freq2, structure[j]], axis=1))
                y_pred3 = model(tf.concat([freq3, structure[j]], axis=1))
                loss = obj_func(y_pred1) + obj_func(y_pred2) + obj_func(y_pred3)
                # loss = obj_func(y_pred2)
            minLoss[j] = tf.reduce_min(loss).numpy()
            minIndex[j] = tf.argmin(loss).numpy()
            minS[j] = structure[j][minIndex[j]].numpy()
            grads = tape.gradient(loss, structure[j])
            opt.apply_gradients(grads_and_vars=zip([grads], [structure[j]]))
            structure[j] = check(structure[j])
            # nega_place = tf.where(structure[j] < 0)
            # structure[j] = tf.Variable(tf.tensor_scatter_nd_update(structure[j], [nega_place], [np.random.uniform(mmin[nega_place[:,1]], mmax[nega_place[:,1]], (nega_place.shape[0]))]))
            # structure[j] = tf.Variable(tf.tensor_scatter_nd_update(structure[j], [nega_place], [np.random.uniform(0, 1, (nega_place.shape[0]))]))
    if np.min([minLoss[0], minLoss[1]]) < bestLoss:
        if minLoss[0] < minLoss[1]:
            bestLoss = minLoss[0]
            bestStructure = minS[0]
            # bestStructure = data_loader.mmX.inverse_transform([minS[0]])[0]
            print(minIndex[0])
        else:
            bestLoss = minLoss[1]
            bestStructure = minS[1]
            # bestStructure = data_loader.mmX.inverse_transform([minS[1]])[0]
            print(num_nodes + minIndex[1])
        print(i, bestLoss)
        print(bestStructure)
        print()

"""