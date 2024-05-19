#%%
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
model.save('dependencies/word2vec-google-news-300.model')
model.save_word2vec_format('dependencies/GoogleNews-vectors-negative300.bin', binary=True)

# %%
