#%%import gensim.downloader as api
from gensim.models import KeyedVectors
import pandas as pd
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean, pdist, squareform
from itertools import combinations
import json
mname="glove-wiki-gigaword-300"#"conceptnet-numberbatch-17-06-300"
model = KeyedVectors.load_word2vec_format(f'dependencies/{mname}.modelwv',binary=True)
alt_data=pd.read_csv('./dependencies/freq.csv')
alt_data=alt_data.set_index("lemma")
alt_data.index.name='words'
word_list=list(model.key_to_index.keys())
common_word_list_noun=alt_data[(alt_data["PoS"]=="n") & alt_data.index.isin(word_list)].index.to_list()
common_word_list=alt_data[alt_data.index.isin(word_list)].index.to_list()

def getv(w):
    return(model[w])
subset_vectors = {word: getv(word) for word in common_word_list}
restricted_model = KeyedVectors(vector_size=model.vector_size)
restricted_model.add_vectors(list(subset_vectors.keys()), list(subset_vectors.values()))

# %%
vectors=[getv(w) for w in common_word_list_noun]
cosine_distances =pdist(vectors,metric='cosine')
cosine_distance_matrix = squareform(cosine_distances)
# %%
word_pairs = list(combinations(common_word_list_noun, 2))

# Combine word pairs with their distances
cosine_df = pd.DataFrame(word_pairs, columns=['Word1', 'Word2'])
cosine_df['CosineDistance'] = cosine_distances
# cosine_df[cosine_df['CosineDistance']<1]
cosine_df_sorted = cosine_df.sort_values(by='CosineDistance', ascending=True).reset_index(drop=True)
cosine_df_sorted["CosineDistance"]=np.round(cosine_df_sorted["CosineDistance"],4)
# with open('output.json', 'w') as file:
#     file.write(json.dumps(cosine_df_sorted[1:70000].values.tolist()))
# %%
# g=cosine_df_sorted[:70000].groupby("Word1").agg({"Word2":list,"CosineDistance":list}).reset_index()
# g['minD']=g["CosineDistance"].apply(min)
# g['maxD']=g["CosineDistance"].apply(max)
# g=g.sort_values(by='minD', ascending=True).reset_index(drop=True)
# # nested_list = g.apply(lambda row: [row['Word1'], row["minD"], row["maxD"],row['Word2'], row['CosineDistance']], axis=1).tolist()

# nested_list = [list(g.Word1),list(g.Word2),list(g.minD),list(g.maxD),list(g.CosineDistance)]
# # %%

# with open('output.json', 'w') as file:
    # file.write(json.dumps(nested_list))
# %%
g=cosine_df_sorted[:500000]
nested_list = [list(g.Word1),list(g.Word2),list(g.CosineDistance)]
with open('output.json', 'w') as file:
    file.write(json.dumps(nested_list))
# %%
