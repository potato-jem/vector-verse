#%%
#from gensim.test.utils import lee_corpus_list
import gensim.downloader as api
from gensim.models import KeyedVectors
import pandas as pd
from random import sample
import numpy as np
import matplotlib.pyplot as plt

#%%
# model = api.load('word2vec-google-news-300')
# model.save('dependencies/word2vec-google-news-300.model')
# model.save_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin', binary=True)
# %%
#model = KeyedVectors.load_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin',binary=True)
model = KeyedVectors.load_word2vec_format('dependencies/glove-wiki-gigaword-300.modelwv',binary=True)
# %%
# x=goog_wordvecs=KeyedVectors.load_word2vec_format('dependencies/word2vec-google-news-300.model',binary=True,limit=1000000)
# %%
animals = ['penguin','lion','panda','lizard','fish','shark','beetle','mouse','bat','ant','sword','knife']
emotions=["virtuous","greed", "anger", "sympathy",'compassion', "love", "hate", "pride", "sloth",
          "agression", "hitler", "pope", "america", "rich", "poor", "capitalist",
         "envy", "wrath", "lust", "gluttony", "murder", "charity", "money", "god", "jesus", "bible","sinful"]
w1=['knife', 'feather', 'cheetah', 'turtle', 'apple', 'banana', 'car', 'bicycle']
planets=["sun","mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
alt_data=pd.read_csv('./dependencies/freq.csv')
alt_data=alt_data.set_index("lemma")
alt_data.index.name='words'
word_list=list(model.key_to_index.keys())
common_word_list_noun=alt_data[(alt_data["PoS"]=="n") & alt_data.index.isin(word_list)].index.to_list()
common_word_list=alt_data[alt_data.index.isin(word_list)].index.to_list()

n=10
w=sample(common_word_list_noun[:100],n)
w=animals

#%%
v1_1="sharp"
v1_2="blunt"
v2_1="big"
v2_2="small"

# Calculate difference vectors
v1 = model[v1_1] - (model[v1_2] if v1_2!="" else 0)
v2 = model[v2_1] - (model[v2_2] if v2_2!="" else 0)
#v2 = model['fast']+0- model['slow']

# v1_sum= (
#      (model["large"]-model["small"])/np.linalg.norm(model["large"]-model["small"]) +
#      (model["giant"]-model["tiny"])/np.linalg.norm(model["giant"]-model["tiny"]) + 
#      (model["bigger"]-model["big"])/np.linalg.norm(model["bigger"]-model["big"]) + 
#      (model["small"]-model["smaller"])/np.linalg.norm(model["small"]-model["smaller"]) +
#      (model["big"]-model["little"])/np.linalg.norm(model["big"]-model["little"])
# )
# v1= v1_sum/np.linalg.norm(v1_sum)

# Normalize the basis vectors
v1_norm = v1/np.linalg.norm(v1)
v2_norm = v2/np.linalg.norm(v2)

def project_onto_basis(word_vector, basis1, basis2):
    # Project the word vector onto the new basis
    projection_1 = np.dot(word_vector, basis1)
    projection_2 = np.dot(word_vector, basis2)
    return projection_1, projection_2

# List of words to project
words_to_project = w

# Project each word and store the results
projections = []
min_axis=1
max_axis=-1
for word in words_to_project:
    word_vector = model[word]
    x_value, y_value = project_onto_basis(word_vector, v1_norm, v2_norm)
    projections.append((word, x_value, y_value))
    min_axis=min(min_axis,min(y_value,x_value))
    max_axis=max(max_axis,max(y_value,x_value))

# Plot the projections
plt.figure(figsize=(10, 8))
for word, x_value, y_value in projections:
    plt.scatter(x_value, y_value)
    plt.text(x_value, y_value, word, fontsize=12)
plt.plot(np.arange(min_axis,max_axis,0.05),np.arange(min_axis,max_axis,0.05))
plt.xlabel(v1_2+" ->"+v1_1)
plt.ylabel(v2_2+" ->"+v2_1)
plt.plot()
plt.title('Word Projections on x_value and y_value Axes')
plt.grid(True)
plt.show()

#%%
subset_vectors = {word: model[word] for word in common_word_list if word in model}
restricted_model = KeyedVectors(vector_size=model.vector_size)
restricted_model.add_vectors(list(subset_vectors.keys()), list(subset_vectors.values()))

# %%
def expl(word,direction,topn=5,weight=1,restricted=True):
    model_to_use=restricted_model if restricted else model
    more_this_direction=model_to_use.most_similar(model[word]+weight*direction,topn=topn)
    less_this_direction=model_to_use.most_similar(model[word]-weight*direction,topn=topn)
    direction1=model_to_use.most_similar(direction,topn=1)[0][0]
    direction2=model_to_use.most_similar(-direction,topn=1)[0][0]
    newlinelist='\n'.join([t[0]+" ("+str(round(t[1],2))+")" for t in more_this_direction[1:]])
    print(f"More in {direction1} direction: \n{newlinelist}")
    newlinelist='\n'.join([t[0]+" ("+str(round(t[1],2))+")" for t in less_this_direction[1:]])
    print(f"\nMore in -{direction1} ({direction2}) direction : \n{newlinelist}")


# %%
#expl("Uranus",v1)
expl("sun",v1,topn=10,weight=1,restricted=True)

# %%
