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
#model = KeyedVectors.load_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin',binary=True)
mname="glove-wiki-gigaword-300"#"conceptnet-numberbatch-17-06-300"
model = KeyedVectors.load_word2vec_format(f'dependencies/{mname}.modelwv',binary=True)
#english_keys = [key for key in original_model.key_to_index.keys() if key.startswith('/c/en/')]
#model = KeyedVectors(vector_size=original_model.vector_size)
#english_vectors = np.array([original_model[key] for key in english_keys])
#evalues=[original_model[key] for key in english_keys]
#ekeys=[key[6:] for key in english_keys]
#model.add_vectors(ekeys,evalues)
#model.update(new_keys)
# for key in english_keys:
#     model.add_vector(key[:5], original_model[key])
# %%
# x=goog_wordvecs=KeyedVectors.load_word2vec_format('dependencies/word2vec-google-news-300.model',binary=True,limit=1000000)
# %%
def getv(w):
    return(model[w])
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
#word_list=[k[6:] for k in model.key_to_index.keys() if k[:5]=='/c/en']
common_word_list_noun=alt_data[(alt_data["PoS"]=="n") & alt_data.index.isin(word_list)].index.to_list()
common_word_list=alt_data[alt_data.index.isin(word_list)].index.to_list()

n=10
w=sample(common_word_list_noun[:100],n)
w=animals

subset_vectors = {word: getv(word) for word in common_word_list}
restricted_model = KeyedVectors(vector_size=model.vector_size)
restricted_model.add_vectors(list(subset_vectors.keys()), list(subset_vectors.values()))

#%%
    #return(model[f"/c/en/{w}"])

v1_1="sharp"
v1_2="blunt"
v2_1="horse"
v2_2="pony"

# Calculate difference vectors
v1 = getv(v1_1) - (getv(v1_2) if v1_2!="" else 0)
v2 = getv(v2_1) - (getv(v2_2) if v2_2!="" else 0)


# v1_sum= (
#      (getv("large")-getv("small"))/np.linalg.norm(getv("large")-getv("small")) +
#      (getv("giant")-getv("tiny"))/np.linalg.norm(getv("giant")-getv("tiny")) + 
#      (getv("bigger")-getv("big"))/np.linalg.norm(getv("bigger")-getv("big")) + 
#      (getv("small")-getv("smaller"))/np.linalg.norm(getv("small")-getv("smaller")) +
#      (getv("big")-getv("little"))/np.linalg.norm(getv("big")-getv("little"))
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
    word_vector = getv(word)
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


# %%
def expl(word,direction,topn=5,weight=1,restricted=True,norm=True):
    model_to_use=restricted_model if restricted else model
    m=getv(word)+weight*direction
    l=getv(word)-weight*direction
    more_this_direction=model_to_use.most_similar(m/(np.linalg.norm(m) if norm else 1),topn=topn)
    less_this_direction=model_to_use.most_similar(l/(np.linalg.norm(l) if norm else 1),topn=topn)
    direction1=model_to_use.most_similar(direction,topn=1)[0][0]
    direction2=model_to_use.most_similar(-direction,topn=1)[0][0]
    newlinelist='\n'.join([t[0]+" ("+str(round(t[1],2))+")" for t in more_this_direction[1:]])
    print(f"More in {direction1} direction: \n{newlinelist}")
    newlinelist='\n'.join([t[0]+" ("+str(round(t[1],2))+")" for t in less_this_direction[1:]])
    print(f"\nMore in -{direction1} ({direction2}) direction : \n{newlinelist}")


# %%
#expl("Uranus",v1)
expl("bat",v2,topn=5,weight=4,restricted=True,norm=True)

# %%
x=getv("elephant")-getv("mouse")

restricted_model.most_similar(getv("sardine")+x,topn=10)
# %%
pairs = [
#     ("continent", "country"),
# ("country", "city"),
# ("city", "neighborhood"),
# ("neighborhood", "street"),
# ("street", "house"),
# ("ocean", "sea"),
# ("sea", "lake"),
# ("lake", "pond"),
# ("pond", "puddle"),
# ("mountain", "hill"),
# ("hill", "mound"),
# ("planet", "continent"),
# ("continent", "island"),
# ("island", "rock"),
# ("forest", "grove"),
# ("grove", "tree"),
# ("tree", "branch"),
# ("branch", "twig"),
# ("twig", "leaf"),
# ("leaf", "vein"),
("book", "chapter"),
("chapter", "page"),
("page", "paragraph"),
("paragraph", "sentence"),
("sentence", "word"),
("word", "letter")#,
# ("river", "stream"),
# ("stream", "brook"),
# ("brook", "creek"),
# ("building", "room"),
# ("room", "closet"),
# ("closet", "drawer"),
# ("drawer", "box"),
# ("box", "packet"),
# ("packet", "envelope"),
# ("envelope", "sheet"),
# ("sheet", "note"),
# ("note", "line"),
# ("line", "dot"),
# ("universe", "galaxy"),
# ("galaxy", "star"),
# ("star", "planet"),
# ("planet", "moon"),
# ("moon", "crater"),
# ("crater", "hole"),
# ("vehicle", "car"),
# ("car", "seat"),
# ("seat", "cushion"),
# ("cushion", "thread"),
# ("thread", "fiber")
]
from sklearn.decomposition import PCA
difference_vectors  = [getv(big) - getv(small) for big, small in pairs]
X = np.stack(difference_vectors)
pca = PCA(n_components=1)
pca.fit(X)
# The first principal component
larger_vector_pca = pca.components_[0]
larger_vector_pca = larger_vector_pca / np.linalg.norm(larger_vector_pca)
# %%
w="page"
print(restricted_model.most_similar((getv(w)/ np.linalg.norm(getv(w))+larger_vector_pca),topn=10))
print(model.most_similar(getv(w)+larger_vector_pca,topn=10))

# %%
def cosine_similarity(x, y):
    # if np.ndim(x)==1:
    #     x=x.reshape(1,-1)
    # if np.ndim(y)==1:
    #     y=y.reshape(1,-1)
    dot_product = np.dot(y,x.T)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity
#%%
w="page"
print(restricted_model.most_similar((getv(w)/ np.linalg.norm(getv(w))),topn=10))
print(model.most_similar(getv(w),topn=10))
#%%
cosine_similarity(getv("sun"),getv("tennis"))

# %%
expl("sun",(getv("tennis")+getv("sun")/2),topn=10)
# %%
w1="sun"
w2="ocean"
#w3="port"
w3="ship"
# print(cosine_similarity(getv(w1),getv(w2)))
# print(cosine_similarity(getv(w2),getv(w3)))
# print(cosine_similarity(getv(w3),getv(w4)))
# print(cosine_similarity(getv(w1),getv(w4)))
print(model.distance(w1,w2))
print(model.distance(w2,w3))
print(model.distance(w1,w2)+model.distance(w2,w3))
# print(model.most_similar((getv(w1)),topn=10))
# print(model.most_similar((getv(w2)),topn=10))
# print(model.most_similar((getv(w1)+getv(w2)),topn=10))
# print(model.most_similar(getv(w2)-(getv(w1)),topn=10))
# print(model.most_similar(getv(w1)-(getv(w2)),topn=10))
#%%
# x=restricted_model.most_similar(getv(w1),topn=1000)
# y=restricted_model.most_similar(getv(w1)-getv(w2),topn=1000)
# z=pd.DataFrame(x,columns=["name","value1"]).merge(pd.DataFrame(y,columns=["name","value2"]),on="name")
# z["value3"]=z["value2"]-z["value1"]
# z=z.sort_values(by="value3")
# print(z)
# #z
# %%
model_to_use=restricted_model#model
d1=1-cosine_similarity(getv(w1),model_to_use.vectors)#model.distances(w1,model.index_to_key)
d2=1-cosine_similarity(getv(w3),model_to_use.vectors)#model.distances(w3,model.index_to_key)
combined_distance=d1+d2+np.abs(d1-d2)
combined_distance[np.isnan(combined_distance)] = np.inf
if w3 in model_to_use.index_to_key:
    combined_distance[model_to_use.key_to_index[w3]]= np.inf
if w1 in model_to_use.index_to_key:
    combined_distance[model_to_use.key_to_index[w1]]= np.inf
closest_index = np.argsort(combined_distance)[:5]
bottom_n_words = [model_to_use.index_to_key[idx] for idx in closest_index]
print(bottom_n_words)
# %%
# %%

# %%
