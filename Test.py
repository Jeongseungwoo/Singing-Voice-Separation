import tensorflow as tf
import librosa
from librosa.util import find_files
from librosa import load

from util import *
from config import *
from U_net import U_Net


def main() :
    music_path = "./sample.wav"
    
    mix_wav_mag, mix_wav_phase = LoadAudio(music_path)

    START = 60
    END = START + patch_size  # 11 seconds

    mix_wav_mag=mix_wav_mag[:, START:END]
    mix_wav_phase=mix_wav_phase[:, START:END]
    
    X = mix_wav_mag[1:].reshape(1,512,128,1)
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag":X},y = None,num_epochs = 1,shuffle = False)

    
    deep_u_net = tf.estimator.Estimator(model_fn=U_Net,model_dir="./model")
    predictions = list(deep_u_net.predict(input_fn=predict_input_fn))
    mask = predictions[0]['outputs']
    
    target_pred_mag = np.vstack((np.zeros((128)), mask.reshape(512, 128)))
    SaveAudio(music_path[:-4]+"_vocal.wav",target_pred_mag,mix_wav_phase)

if __name__ == "__main__" : 
    main()