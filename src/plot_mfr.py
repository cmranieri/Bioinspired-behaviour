import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
import pickle


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    with open( '../mfr/marmoset/mfr_str/cnn-lstm/fd0_s1_lbl1.pickle', 'rb' ) as f:
        sample = pickle.load( f )
    sample = sample[ :, 500:700 ]
    print( sample.shape )

    matplotlib.rcParams['xtick.labelsize']='large'
    matplotlib.rcParams['ytick.labelsize']='large'
    fig, axs = plt.subplots( 4, 1, figsize=(9,5) )
    titles = [ 'Ctx_RS_1', 'Ctx_RS_2', 'Ctx_FSI_1', 'Ctx_FSI_2' ]
    for i in range( 4 ):
        sample_ch = sample[ i ]

        #sample_ch = butter_bandpass_filter( sample_ch, 8, 50, 1000 )
        #sample_ch = np.reshape( sample_ch, [-1,1] )
        #scaler = StandardScaler()
        #sample_ch = scaler.fit_transform( sample_ch )

        axs[i].plot( sample_ch )
        axs[i].set_ylabel( '%s\n[a.u.]' % titles[i], fontsize='large' )
        axs[i].set_xlim( 0, len(sample[i]) )
        axs[i].set_ylim( np.min( sample_ch ), np.max( sample_ch ) )
        axs[i].set_xticks( [] )
        axs[i].set_yticks( [] )
        #axs[i].ticklabel_format( style='sci', scilimits=(0,0) )
    axs[3].set_xticks( [ 0, 50, 100, 150, 200 ] )
    axs[3].set_xlabel( 'Timestep ($\\times 20$ ms)', fontsize='large' )
    fig.tight_layout()
    plt.subplots_adjust( hspace = 1.5 )
    plt.savefig( '../images/sample.pdf' )
    plt.close()





