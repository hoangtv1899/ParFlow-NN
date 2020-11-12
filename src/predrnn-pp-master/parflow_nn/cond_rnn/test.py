import numpy as np
import tensorflow as tf

from cond_rnn.cond_rnn import ConditionalRNN

NUM_SAMPLES = 10
TIME_STEPS = 720
INPUT_DIM = 1
NUM_CELLS = 24
COND_1_DIM = 20
COND_2_DIM = 30
NUM_CLASSES = 8

def create_conditions(input_dim):
    return np.array([np.random.choice([0, 1], size=input_dim, replace=True) for _ in range(NUM_SAMPLES)], dtype=float)


class MySimpleModel(tf.keras.Model):
    def __init__(self):
        super(MySimpleModel, self).__init__()
        self.cond = ConditionalRNN(NUM_CELLS, cell='LSTM', dtype=tf.float32, return_sequences = True)
        self.out = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = self.out(o)
        return o


train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
train_cond_1 = create_conditions(input_dim=COND_1_DIM)
train_cond_2 = create_conditions(input_dim=COND_2_DIM)

model = MySimpleModel()
tt = model.cond([train_inputs, train_cond_1, train_cond_2])
print(tt.shape)
