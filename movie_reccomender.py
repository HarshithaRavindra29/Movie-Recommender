# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:43:06 2018

@author: Harshitha R

Movie recommendation using Movie Lens data
"""


import pandas as pd

#Using apyori library to find relation between movies using support metric
from apyori import apriori  


ratings = pd.read_csv("ratings.csv")
ratings = ratings.drop(columns = ['timestamp'])

ratings.isnull().sum()
ratings.describe()

ratings.movieId = ratings.movieId.astype(str)
ratings.userId = ratings.userId.astype(str)
ratings.describe()

"""Seperating good bad and ok movies
A movie with highg rating is expectted to appear with other high rated movie
Similarly a low or ok rated movies are grouped together to check for association between them
"""
gRating = ratings[ratings.rating >= 4]
bRating = ratings[ratings.rating <= 2]
bRating = bRating.reset_index()
bRating.drop(['index'] ,axis = 1, inplace = True)

oRating = ratings[ratings.rating > 2][ratings.rating < 4]
oRating = oRating.reset_index()
oRating.drop(['index'] ,axis = 1, inplace = True)

def find_association(dataset):
    watch_dict = {}
    for i in dataset.userId.unique():
        movies = [dataset[dataset.userId==i].movieId.unique()]
        watch_dict.update({i:movies})
        
    watch_df = pd.DataFrame(watch_dict).transpose()
    watch_df.columns = ['movies']
    watch_df['userId'] = watch_df.index.astype(str)
    
    mov_list = watch_df.movies.tolist()
    association_results = list(apriori(mov_list))
    if len(association_results) > 0:
        rec_outcome = pd.DataFrame()
        for mov in association_results:
            pair = mov[0] 
            movs = [x for x in pair]
            rec_outcome = rec_outcome.append({'Combination':movs,'Support':mov[1],'Jkey':'-'.join(movs)}, ignore_index = True)    
        Use_set = rec_outcome[rec_outcome.Combination.str.len() > 1]
    else:
        Use_set = pd.DataFrame({'Combination':[],'Support':0,'Jkey':'-'})
    return Use_set

good_movies = find_association(gRating)
bad_movies = find_association(bRating)
ok_movies = find_association(oRating)

"""
Using only support metric to find association. 
Hypothesis: highly rated movie combination would not occur in low rated movie list
if it occurs, the the support is subracted and the score becomes more negative/smaller
Making the system not recommend that combination
"""

all_combined = pd.merge(good_movies,bad_movies,how = 'outer',on = ['Jkey'])
all_combined = pd.merge(all_combined,ok_movies,how = 'outer',on = 'Jkey')
all_combined.fillna(0, inplace=True)
all_combined['Rec_score'] = (all_combined.Support_x-all_combined.Support_y-all_combined.Support)

Recommender =  all_combined[['Combination_x','Rec_score']]  
"""
Based on the Rec_score, movie is recommended in order of score and if the user has not 
watched the recommended movie yet

Code can be improved further when no movie combination has enough support to be recommended
""" 