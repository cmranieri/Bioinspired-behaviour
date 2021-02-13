import pickle
import os
import re

os.chdir('../data')
for fname in os.listdir():
    if not re.match( '.*pickle.*', fname ):
        continue
    with open( fname, 'rb' ) as f:
        x = pickle.load( f )
    with open( fname, 'wb' ) as f:
        pickle.dump( x, f, protocol=0 )
