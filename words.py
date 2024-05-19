#%%
import numpy as np
import pandas as pd
from random import sample
import plotly.graph_objects as go

def drop_prefix(self, length):
    self.index = self.index.str[length:]
    return self

pd.core.frame.DataFrame.drop_prefix = drop_prefix
pd.options.display.max_rows = 50


wordEmbeddingAll = pd.read_hdf('./dependencies/mini.h5')
wEmb = wordEmbeddingAll.filter(like='/c/en/', axis=0).drop_prefix(len('/c/en/'))

data = pd.read_csv('./dependencies/wordList.txt', sep=" ", header=None)
data=data.drop_duplicates()
data=data.set_index(0)
data.index.name='words'
common_word_list=data.join(wEmb,how='inner').index.to_list()

alt_data=pd.read_csv('./dependencies/freq.csv')
alt_data=alt_data.set_index("lemma")
alt_data.index.name='words'
common_word_list_noun=alt_data[alt_data["PoS"]=="n"].join(wEmb,how='inner').index.to_list()

w_norm1=(wEmb-wEmb.min(axis=0))/(wEmb.max(axis=0)-wEmb.min(axis=0))
w_norm2=wEmb/np.maximum(-wEmb.min(axis=0),wEmb.max(axis=0))
w_norm3=np.maximum(np.minimum(w/30,1),-1)
#%%
def rankWords2(df,wordList,word1,word2):
    newList = wordList[:]
    #newList.append(word1)
    #newList.append(word2)
    a_minus_b=df.loc[word1] - df.loc[word2]
    return norm_and_rank_vector(df.loc[newList]-df.loc[word2],a_minus_b)

def norm_and_rank_vector(v1,v2):
    if type(v1)==pd.Series:
        v1_norm=v1/np.sqrt(np.square(v1).sum())
    else:
        v1_norm=v1.divide(np.sqrt(np.square(v1).sum(axis=1)),axis=0).fillna(0)
    v2_norm=v2/np.sqrt(np.square(v2).sum())
    zero_to_1_scale=((v1_norm.dot(v2_norm))/v2_norm.dot(v2_norm)).dropna().sort_values()
    return 2*zero_to_1_scale-1

def randList(n):
    return sample(common_word_list,n)

def rankWords1(df,wordList,word1):
    newList = wordList[:]
    #newList.append(word1)
    return norm_and_rank_vector(df.loc[newList],df.loc[word1])
    
# %%
# print(w_norm1.loc[['hot','cold']])
animals = ['penguin','polar_bear','lion','panda','lizard','fish','shark','beetle','mouse','bat','ant','sword','knife']
emotions=["virtuous","greed", "anger", "sympathy",'compassion', "love", "hate", "pride", "sloth",
          "agression", "hitler", "pope", "america", "rich", "poor", "capitalist",
         "envy", "wrath", "lust", "gluttony", "murder", "charity", "money", "god", "jesus", "bible","sinful"]
# print(w_norm1.loc[emotions].dot(w_norm1.loc["good"] - w_norm1.loc["bad"]).sort_values())
# %%
#rankWords2(w_norm1,common_word_list,'sharp','soft')#.to_clipboard()
# print(rankWords1(w_norm1,animals,'sharp'))
# print(rankWords1(w_norm1,animals,'meaningful'))
n=20
w1,w2='soft','sharp'
w3,w4='boring','fun'
w=sample(common_word_list_noun[:100],n)
# x=rankWords1(w_norm1,w,w1)
# y=rankWords1(w_norm1,w,w2)
x=rankWords2(w_norm2,w,w1,w2)
y=rankWords2(w_norm2,w,w3,w4)
#print(rankWordsRand(df,'sharp','soft',5))

fig = go.Figure()

# Add scatter trace
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers+text',  # Show markers and text
    text=w,  # Use words as text labels
    textposition='top center',  # Position text labels
    marker=dict(size=12, color='skyblue'),  # Customize marker style
))

# Customize layout
fig.update_layout(
    title='Scatter Plot with Words as Labels',
    xaxis=dict(title=w1+"->"+w2),
    yaxis=dict(title=w3+"->"+w4)
)

# Show the plot
fig.show()
#%%

# word1='sharp'
# word2='soft'
# df=w_norm1
# newList=animals
# a_minus_b=(df.loc[word1] - df.loc[word2])
# a_minus_b=a_minus_b/np.sqrt(np.square(a_minus_b).sum())
# x=df.loc[newList]-df.loc[word2]
# x.divide(np.sqrt(np.square(x).sum(axis=1)),axis=0)
# np.sqrt(np.square(x).sum(axis=1))
# ((x.dot(a_minus_b)/np.sqrt(np.square(x).sum(axis=1)))/a_minus_b.dot(a_minus_b)).dropna().sort_values()

# v1=df.loc[newList]
# z=v1.divide(np.sqrt(np.square(v1).sum(axis=1)),axis=0)

# # %%
# newList = animals[:]
# newList.append('sharp')
# newList.append('soft')
# a_minus_b=df.loc['sharp'] - df.loc['soft']
# v1=df.loc[newList]-df.loc['soft']
# v2=a_minus_b
# v1_norm=(v1).divide(np.sqrt(np.square((v1)).sum(axis=1)),axis=0).fillna(0)
# v2_norm=(v2)/np.sqrt(np.square((v2)).sum())
# ((v1_norm.dot(v2_norm))/v2_norm.dot(v2_norm)).dropna().sort_values()

# np.sqrt(np.square((2*v1-1)).sum(axis=1))

# %%
