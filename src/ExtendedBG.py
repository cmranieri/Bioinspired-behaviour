from GenericBG import Network
import numpy as np
from netpyne import sim
import pickle
from utils import compute_mfr
import matplotlib.pyplot as plt



class ExtendedBG( Network ):
    def __init__( self, stim_interval, preds, **kwargs ):
        super( ExtendedBG, self ).__init__( n_channels = 2,
                                            **kwargs )
        self.stim_interval = stim_interval
        self.preds = preds
        self.W = self._initialize_weights()
        self.stim_c0 = 0
        self.stim_c1 = 0
        self._tt = 0


    def _initialize_weights( self ):
        # [ cereals, tidy, laptop, newspaper, sandwich, smartphone, table, tea, dishes ]
        W = [ [ 1, 0, 0, 0, 1, 0, 0, 1, 0 ],
              [ 0, 0, 1, 1, 0, 1, 0, 0, 0 ] ]
        return np.array( W, dtype=np.float32 )


    def get_next_input( self ):
        t = self._tt % 140
        X = self.preds[ t ] 
        self._tt += 1
        return np.array( X )


    def _stimSim( self, t ):
        # < 1.0e-3
        X = self.get_next_input()
        self.stim_c0 = np.sum( (X * self.W)[0] ) * 1.0e-3
        self.stim_c1 = np.sum( (X * self.W)[1] ) * 1.0e-3
        print( 'Stims: %0.4g, %0.4g' % (self.stim_c0, self.stim_c1) )
        #sim.net.modifyStims({'conds': {'source': 'Input_th_%d'%0}, 'del': 0, 'amp': self.stim_c0 })
        #sim.net.modifyStims({'conds': {'source': 'Input_th_%d'%1}, 'del': 0, 'amp': self.stim_c1 })
        sim.net.modifyStims({'conds': {'source': 'Input_StrD1_%d'%0}, 'del': 0, 'amp': self.stim_c0 })
        sim.net.modifyStims({'conds': {'source': 'Input_StrD1_%d'%1}, 'del': 0, 'amp': self.stim_c1 })
        sim.net.modifyStims({'conds': {'source': 'Input_StrD2_%d'%0}, 'del': 0, 'amp': self.stim_c0 })
        sim.net.modifyStims({'conds': {'source': 'Input_StrD2_%d'%1}, 'del': 0, 'amp': self.stim_c1 })


    def get_mfr( self, bins=50 ):
        all_spikes = self.extractSpikes()
        mfr = compute_mfr( all_spikes, self._tt*500, bins=bins )
        #plt.plot( mfr[ 'Cor_RS_APs_0' ], color='blue' )
        #plt.plot( mfr[ 'Cor_RS_APs_1' ], color='red' )
        #plt.savefig( '../images/mfr.png' )
        #plt.clf()
        mfr_list = [ mfr[key] for key in sorted( mfr ) ]
        print( sorted(mfr.keys()) )
        return np.array( mfr_list )


    def get_lfp( self ):
        lfp = self.extractLFP_raw()
        #plt.plot( np.transpose( lfp, [1,0] ) )
        #plt.savefig( '../images/lfp.png' )
        #plt.clf()
        return lfp


    # Override
    def simulate( self, dt=0.1, lfp=False, recordStep=1, seeds=None ):
        simConfig = self.buildSimConfig(dt=dt, lfp=lfp, recordStep=recordStep, seeds=seeds)
        sim.initialize(                     # create network object and set cfg and net params
                simConfig = simConfig,      # pass simulation config and network params as arguments
                netParams = self.netParams)
        sim.net.createPops()                # instantiate network populations
        sim.net.createCells()               # instantiate network cells based on defined populations
        sim.net.connectCells()              # create connections between cells based on params
        sim.net.addStims()                  # add stimulation
        sim.setupRecording()                # setup variables to record for each cell (spikes, V traces, etc)
        sim.runSimWithIntervalFunc( self.stim_interval, self._stimSim )
        sim.gatherData()                    # gather spiking data and cell info from each node
        sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
        #sim.analysis.plotData()             # plot spike raster


if __name__ == '__main__':
    network = ExtendedBG( stim_interval = 500,
                          t_sim = t_sim,
                          has_pd = False )
    network.simulate( recordStep=10, lfp=True )
    network.save_spikes()
    network.save_lfp()
    #print( sim.allSimData['popRates'] )
    #print( sim.allSimData['avgRate'] )
    #print( len(sim.allSimData['spkid']), len(sim.allSimData['spkt']) )
