from main import compare_convergences
import numpy as np
import pandas as pd

YEAR = 2023
nfl_df = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
                   'play_by_play_' + str(YEAR) + '.csv.gz',
                   compression= 'gzip', low_memory= False)
data = nfl_df[['passing_yards','epa']].dropna().to_numpy()
non_adaptive = compare_convergences([(0,0,1),(0,10,2),(0,20,4),(0,30,6),(0,40,8),(0,50,10)], data, 1000, False, '../results/applied_study/')
adaptive = compare_convergences([(0,0,1),(0,10,2),(0,20,4),(0,30,6),(0,40,8),(0,50,10)], data, 1000, True, '../results/applied_study/')