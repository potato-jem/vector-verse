#%%
import gensim.downloader as api
model_name="conceptnet-numberbatch-17-06-300"#"fasttext-wiki-news-subwords-300"#"glove-wiki-gigaword-50"#"glove-wiki-gigaword-300"#'word2vec-google-news-300'#glove-twitter-200 
output_name=model_name
model = api.load(model_name)
#model.save('dependencies/'+model_name+'.model')
model.save_word2vec_format('dependencies/'+model_name+'.modelwv', binary=True)

# %%
