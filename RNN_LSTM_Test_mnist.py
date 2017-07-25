import numpy as np
np.random.seed(1337)
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import LSTM,TimeDistributed,Dense
from keras.optimizers import Adam

BATCH_START=0
TIME_STEPS=20
BATCH_SIZE=50
INPUT_SIZE=1
OUTSIZE=1
CELL_SIZE=20
LR=0.006

def get_batch():
    global BATCH_START,TIME_STEPS
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))/(10*np.pi)
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START+=TIME_STEPS
    plt.plot(xs[0,:],res[0:],'r',xs[0,:],seq[0,:],'b--')
    plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

get_batch()
exit()