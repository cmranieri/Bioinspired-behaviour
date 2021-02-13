import pickle
import sys
import numpy as np

all_x = { 'results': list(),
          'times':   list() }
correct    = list()
incorrect  = list()
unfinished = list()
for index in range(8):
    path  = '../results/sim_cnn-lstm_%s.pickle'%(index+1)

    with open( path, 'rb' ) as f:
        x = pickle.load( f )

    all_x['results'] += x[ 'results' ]
    all_x['times']   += x[ 'times' ]

    correct.append(     sum( [ a == 'correct'    for a in x['results'] ] ) )
    incorrect.append(   sum( [ a == 'incorrect'  for a in x['results'] ] ) )
    unfinished.append(  sum( [ a == 'unfinished' for a in x['results'] ] ) )

total = np.sum(correct) + np.sum(incorrect) + np.sum(unfinished)
print( 'Correct:',    np.sum(correct)    / total )
print( 'Incorrect:',  np.sum(incorrect)  / total )
print( 'Unfinished:', np.sum(unfinished) / total )

t_correct =   [ all_x['times'][i] for i in range(len(all_x['results'])) if all_x['results'][i]=='correct' ]
t_incorrect = [ all_x['times'][i] for i in range(len(all_x['results'])) if all_x['results'][i]=='incorrect' ]
print( 'T_correct',   np.mean(t_correct),   np.std(t_correct) )
print( 'T_incorrect', np.mean(t_incorrect), np.std(t_incorrect) )
