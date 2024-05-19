#%%
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import gensim.downloader as api
#model = Word2Vec(lee_corpus_list, vector_size=300, epochs=100)
# model = api.load('word2vec-google-news-300')
# model.save('dependencies/word2vec-google-news-300.model')
# model.save_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin', binary=True)
# %%
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin',binary=True)
# %%
# x=goog_wordvecs=KeyedVectors.load_word2vec_format('dependencies/word2vec-google-news-300.model',binary=True,limit=1000000)
# %%
animals = ['penguin','polar_bear','lion','panda','lizard','fish','shark','beetle','mouse','bat','ant','sword','knife']
emotions=["virtuous","greed", "anger", "sympathy",'compassion', "love", "hate", "pride", "sloth",
          "agression", "hitler", "pope", "america", "rich", "poor", "capitalist",
         "envy", "wrath", "lust", "gluttony", "murder", "charity", "money", "god", "jesus", "bible","sinful"]
w1=['knife', 'feather', 'cheetah', 'turtle', 'apple', 'banana', 'car', 'bicycle']
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
words_to_project = animals

# Project each word and store the results
projections = []
minx=1
miny=1
maxx=-1
maxy=-1
for word in words_to_project:
    word_vector = model[word]
    v12d, v22d = project_onto_basis(word_vector, v1, v2)
    projections.append((word, v12d, v22d))
    minx=min(minx,v12d)
    miny=1
    maxx=-1
    maxy=-1

# Plot the projections
plt.figure(figsize=(10, 8))
for word, v12d, v22d in projections:
    plt.scatter(v12d, v22d)
    plt.text(v12d, v22d, word, fontsize=12)
plt.plot(range())
plt.xlabel(v1_2+" ->"+v1_1)
plt.ylabel(v2_2+" ->"+v2_1)
plt.title('Word Projections on v12d and v22d Axes')
plt.grid(True)
plt.show()

# %%
