import pickle
import os
import numpy as np

def fuse( index ):
    v_file = open( '../data/model-lyell-cnn-lstm-0%d.pickle'%index, 'rb' ) 
    i_file = open( '../data/model-lyell-imulstm-0%d.pickle'%index, 'rb' ) 

    wv = 4
    wi = 1

    video_dict = pickle.load( v_file )
    video  = video_dict['predictions']
    labels = video_dict['labels']
    imu    = pickle.load( i_file )['predictions']
    fused  = np.zeros( video.shape )

    v_file.close()
    i_file.close()

    print(video.shape)
    print(imu.shape)

    for i in range( fused.shape[0] ):
        for j in range( fused.shape[1] ):
            fused[i, j] = wv*video[i, j] + wi*imu[i, j]

    with open( '../data/model-lyell-fused-0%d.pickle'%index, 'wb' ) as f:
        pickle.dump( { 'predictions': fused,
                       'labels': labels }, f )


if __name__ == '__main__':
    for i in range(8):
        fuse( i+1 )
