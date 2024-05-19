#%%
#from gensim.test.utils import lee_corpus_list
import gensim.downloader as api
from gensim.models import KeyedVectors
import pandas as pd
from random import sample
#%%
model = api.load('word2vec-google-news-300')
model.save('dependencies/word2vec-google-news-300.model')
model.save_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin', binary=True)
# %%
model = KeyedVectors.load_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin',binary=True)
# %%
# x=goog_wordvecs=KeyedVectors.load_word2vec_format('dependencies/word2vec-google-news-300.model',binary=True,limit=1000000)
# %%
animals = ['penguin','polar_bear','lion','panda','lizard','fish','shark','beetle','mouse','bat','ant','sword','knife']
emotions=["virtuous","greed", "anger", "sympathy",'compassion', "love", "hate", "pride", "sloth",
          "agression", "hitler", "pope", "america", "rich", "poor", "capitalist",
         "envy", "wrath", "lust", "gluttony", "murder", "charity", "money", "god", "jesus", "bible","sinful"]
w1=['knife', 'feather', 'cheetah', 'turtle', 'apple', 'banana', 'car', 'bicycle']
alt_data=pd.read_csv('./dependencies/freq.csv')
alt_data=alt_data.set_index("lemma")
alt_data.index.name='words'
word_list=list(model.key_to_index.keys())
common_word_list_noun=alt_data[(alt_data["PoS"]=="n") & alt_data.index.isin(word_list)].index.to_list()
n=10
w=sample(common_word_list_noun[:100],n)
#%%
import numpy as np
import matplotlib.pyplot as plt

v1_1="sharp"
v1_2="soft"
v2_1="sharp"
v2_2=""

# Calculate difference vectors
v1 = model[v1_1] - (model[v1_2] if v1_2!="" else 0)
v2 = model[v2_1] - (model[v2_2] if v2_2!="" else 0)
#v2 = model['fast']+0- model['slow']

# Normalize the basis vectors
v1 /= np.linalg.norm(v1)
v2 /= np.linalg.norm(v2)

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
    x_value, y_value = project_onto_basis(word_vector, v1, v2)
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

# %%

