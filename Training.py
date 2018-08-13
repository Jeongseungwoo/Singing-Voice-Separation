import tensorflow as tf

from U_net import U_Net
from util import *
from config import *


def train() : 
    
    X_list, Y_list = LoadSpectrogram()# Mix spectrogram
    X_mag,X_phase = Magnitude_phase(X_list)
    Y_mag,_ = Magnitude_phase(Y_list)
    
    deep_u_net = tf.estimator.Estimator(model_fn=U_Net,model_dir="./model")
    
    for e in range(EPOCH) :
        # Random sampling for training
        X,y = sampling(X_mag,Y_mag)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag":X},y = y,batch_size = BATCH,num_epochs = 1,shuffle = False)
    
        deep_u_net.train(input_fn= train_input_fn)
        
        
if __name__ == '__main__' :
    train()
    print("Training Complete!!")