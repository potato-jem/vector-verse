#%%
import openai
import math
import numpy as np
import pandas as pd
from itertools import accumulate
import tiktoken
import string 
client = openai.OpenAI()
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
import nltk
from nltk.corpus import stopwords
#%%
from nltk.corpus import stopwords
# nltk.download('stopwords')
# Get the list of stop words
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(string.punctuation))

word_list=pd.read_csv('./dependencies/freq.csv')

lemma_list=pd.read_csv('./dependencies/lemma.en.txt', delimiter=' -> ')
lemma_list[['lemma', 'cnt']] = lemma_list['lemma'].str.split("/", n=1, expand=True)
lemma_list["cnt"]=pd.to_numeric(lemma_list['cnt'], errors='coerce')
lemma_list["word"]=lemma_list["lemma"]+','+lemma_list["word"]
lemma_list['word'] = lemma_list['word'].str.split(",")
lemma_list = lemma_list.explode('word').reset_index()
lemma_list=lemma_list[lemma_list.cnt>250]
prepositions=["about","above","across","after","against","along","among","around","before","behind","below","beneath","beside","between","beyond","concerning","despite","down","during","except","from","inside","into","like","near","onto","outside","regarding","since","through","throughout","toward","underneath","until","upon"]
extra_rows=pd.DataFrame(prepositions,columns=["lemma"])
extra_rows["word"]=extra_rows["lemma"]
extra_rows["index"]=-1
lemma_list=pd.concat([lemma_list,extra_rows])
lemma_list=lemma_list.set_index("word",drop=False)
word_to_lemma_map = lemma_list[~lemma_list.word.duplicated()].lemma.to_dict()

very_common_words=list(lemma_list[lemma_list["index"]<1000].lemma.unique())
less_common_words=list(lemma_list.word)


word_list=word_list.set_index("lemma")
word_list.index.name='words'
word_list=word_list[~(word_list.index.isin(stop_words) | (word_list["PoS"]=="x"))]
word_list=list(word_list.index)
#%%
def get_example_pairs(input_text,
                      max_total_tokens=8,
                      max_output_tokens=100,
                      temperature=1,
                      top_logprobs=3,
                      system_instruction="Continue the sentence without repeating the prompt",
                      category="",
                      stop=[],
                      n=1):
    
    input_length=len(tokenizer.encode(input_text))
    responseX = client.chat.completions.create(
        model="gpt-4o-mini"
        ,messages=[
            {"role": "system",
            "content": system_instruction},
            {
            "role": "user",
            "content": input_text
            }
            ]
        ,max_tokens=min(max_output_tokens,max(1,max_total_tokens-input_length))
        ,temperature=temperature
        ,logprobs=True
        ,top_logprobs=top_logprobs
        ,n=n
        ,stop=stop
    )
    # print(
    #         {"role": "system",
    #         "content": system_instruction},
    #         {
    #         "role": "user",
    #         "content": input_text
    #         }
    # )
    # print(
    #     min(max_output_tokens,max(1,max_total_tokens-input_length))
    #     ,temperature
    #     ,top_logprobs
    #     ,n
    #     ,stop
    #     )
    fingerprint=responseX.system_fingerprint
    df=pd.DataFrame()
    i=0
    for choice in responseX.choices:
        cumulative_message = input_text+" "
        cumulative_token_length = responseX.usage.prompt_tokens-11-len(tokenizer.encode(system_instruction))
        print(choice.message.content)
        if(choice.message.content==''):
            dfp=pd.DataFrame([""],columns=["full_word"])
            dfp["prob1"]=0
            dfp["prob2"]=0
            dfp["context"] = cumulative_message
            dfp["context_length"] =cumulative_token_length
            dfp["start_text"]= input_text
            dfp["id"] = hash(fingerprint+input_text+str(i))
            dfp["fp"] = fingerprint
            dfp["usage"] = responseX.usage.total_tokens
            dfp["category"]=category
            dfp["full_message"]=choice.message.content
            df=pd.concat([df,dfp])
        else:
            for tkn_info in choice.logprobs.content:
                tokens=[[z.token,z.logprob] for z in tkn_info.top_logprobs]
                dfp=pd.DataFrame(tokens,columns=["token0","logprob"])
                dfp["prob"]=np.exp(dfp["logprob"])
                tokens2 = {f'token{i+1}': dfp['token0'].iloc[i] for i in range(len(dfp))}
                probs = {f'prob{i+1}': dfp['prob'].iloc[i] for i in range(len(dfp))}
                combined_dict = {**tokens2, **probs}
                dfp = pd.DataFrame([combined_dict])
                dfp["cum_prob2"]=dfp["prob1"]+dfp["prob2"]
                dfp["cum_prob3"]=dfp["prob1"]+dfp["prob2"]+dfp["prob3"]
                #dfp["cum_sum"] = list(accumulate(dfp["prob"]))
                #dfp["cum_max"] = max(dfp["prob"])
                dfp["actual_token"] = tkn_info.token
                dfp["context"] = cumulative_message
                dfp["context_length"] =cumulative_token_length
                dfp["start_text"]= input_text
                dfp["id"] = hash(fingerprint+input_text+str(i))
                dfp["fp"] = fingerprint
                dfp["usage"] = responseX.usage.total_tokens
                dfp["category"]=category
                dfp["full_message"]=choice.message.content
                dfp["full_word"]=choice.message.content.split(" ")[0]
                df=pd.concat([df,dfp])
                cumulative_message+=tkn_info.token
                cumulative_token_length+=1
        i+=1
    print(responseX.usage.total_tokens)    
    df=df.reset_index(drop=True)
    
    return(df)

#%%
common_words=[
    "The", "I", "It", "He", "She", "They", "We", "You", "A", "An",
    "This", "That", "There", "What", "When", "Where", "Why", "How",
    "If", "In", "On", "As", "At", "For", "To", "With", "But", "And",
    "So", "While"
]
random_words = [
    "Suddenly",
    "Perhaps",
    "Gently",
    "Silence",
    "Shadows",
    "Tomorrow",
    "Darkness",
    "Beyond",
    "Meanwhile",
    "Instantly",
    "Fortunately",
    "Anger",
    "Quietly",
    "Somewhere",
    "Gracefully",
    "Yesterday",
    "Time",
    "Hesitantly",
    "Slowly",
    "Somewhere",
    "Victory",
    "Strangely",
    "Never",
    "Often",
    "Forever",
    "Betrayal",
    "Hope",
    "Rarely",
    "Dreams",
    "Desperately",
    "Mystery",
    "Comfortably",
    "Painfully",
    "Once",
    "Here",
    "Fear",
    "Overwhelmed",
    "Quiet",
    "Somewhere",
    "Downstairs",
    "Thankfully",
    "Hunger",
    "Pain",
    "Brightness",
    "Eventually",
    "Tenderly",
    "Alone",
    "Yesterday",
    "Trembling",
    "Suddenly"
]

#%%
z=get_example_pairs("It allows for a",temperature=0,n=1)
#%%
df_common_t1=pd.concat([get_example_pairs(x,temperature=1,category="common1_t1",n=50) for x in common_words])
df_common_t05=pd.concat([get_example_pairs(x,temperature=0.5,category="common1_t0.5",n=25) for x in common_words])
df_common_t15=pd.concat([get_example_pairs(x,temperature=1.5,category="common1_t1.5",n=25) for x in common_words])
df_random_t1=pd.concat([get_example_pairs(x,temperature=1,category="random1_t1",n=3) for x in random_words])
df_random_t05=pd.concat([get_example_pairs(x,temperature=0.5,category="random1_t0.5",n=3) for x in random_words])
df_random_t15=pd.concat([get_example_pairs(x,temperature=1.5,category="random1_t1.5",n=3) for x in random_words])
df=pd.concat([df_common_t1,
df_common_t05,
df_common_t15,
df_random_t1,
df_random_t05,
df_random_t15])
#%%
manual_sentences=["The cool evening"]
df_manual=pd.concat([get_example_pairs(x,temperature=1,category="manual",n=5) for x in manual_sentences])
#%%
sentence_starters=[]
#sentence_starters=common_words[1:20]
#df=pd.concat([get_example_pairs(x,temperature=1) for x in sentence_starters])
# df2u=pd.concat([get_example_pairs(x,temperature=1,category="2upper") for x in words_2_upper])
# df2l=pd.concat([get_example_pairs(x,temperature=1,category="2lower") for x in words_2_lower])
# df3u=pd.concat([get_example_pairs(x,temperature=1,category="3upper") for x in words_3_upper])
# df4u=pd.concat([get_example_pairs(x,temperature=1,category="4upper") for x in words_4_upper])
# df4u_business=pd.concat([get_example_pairs(x,temperature=1,category="4upper_business") for x in words_4_upper_business])
# df=pd.concat([df2u,df2l,df3u,df4u,df4u_business])
# df2u.to_csv("outputs/run1.csv",index=False)
#%%
def get_score(prob1,prob2,context_length,pos2=2,k=25,j=0.2,desired_max_length=5,length_weight=0.1):
    score_base=(1 / (1 + np.exp(-k * (prob2 - j))))
    length_adj=(np.exp(-np.maximum(context_length-desired_max_length,0)*length_weight))
    position_score=(6.0-pos2)/4.0
    return 0.33*(score_base)*length_adj+0.33*position_score+0.33*(prob1+prob2)

def get_score_previous(prob2,context_length,k=25,j=0.2,desired_max_length=5,length_weight=0.1):
    score_base=(1 / (1 + np.exp(-k * (prob2 - j))))
    return (score_base)*(np.exp(-np.maximum(context_length-desired_max_length,0)*length_weight))


def filter_results(df):
    df["classification"]=""
    df["token1_stripped"]=df["token1"].str.strip().str.lower()
    df["token2_stripped"]=df["token2"].str.strip().str.lower()
    #df["score_base"]=1 / (1 + np.exp(-25 * (df["prob2"] - 0.2)))
    #df["score"]=(1 / (1 + np.exp(-25 * (df["prob2"] - 0.2))))*np.exp(-np.maximum(df["context_length"]-5,0)*0.1)
    df["score"]=get_score(df["prob1"],df["prob2"],df["context_length"])
    is_stop_word=(df["token1_stripped"].isin(stop_words)) | (df["token2_stripped"].isin(stop_words)) 
    is_short_word=(df["token1_stripped"].str.len()<3) | (df["token2_stripped"].str.len()<3)
    is_short_context=(df["context"].str.len()<8) | (df["context_length"]<=2)
    # potential_condition=(
    #     #(df["cum_prob2"]>0.7)  
    #     # & (df["prob1"]<0.6)  
    #     ~(df["token1_stripped"].isin(stop_words)) 
    #      & ~(df["token2_stripped"].isin(stop_words))  
    #      & (df["token1_stripped"].str.len()>=4) 
    #      & (df["token1_stripped"].str.len()>=4)) 
    df.loc[is_stop_word,"classification"] = "is_stop_word"
    df.loc[is_short_word,"classification"] = "is_short_word"
    df.loc[is_short_context,"classification"] = "is_short_context"
    excl_list=["my ","my,","I ","I'","information","model","intelligence","artificial","data"]
    df_filtered=df[~is_stop_word & ~is_short_word & ~is_short_context].sort_values("score",ascending=False)
    import re
    def has_direct_repeated_words(text):
        # Regular expression to find directly repeated words
        pattern = r'\b(\w+)\s+\1\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    def update_table_with_longer_words(df_original):
        def min_or_alt(s,alt):
                return s.min() if not s.empty else alt
        for word_num in ["1","2"]:
            token_name="token"+word_num
            prob_name="prob"+word_num
            #token_replacement_filterp1=((~df_original[token_name].isin(very_common_words))&(df_original[token_name].str.len()<=3))
            token_replacement_filter=[min_or_alt(lemma_list[lemma_list.word.str.startswith(x)]["index"],100000)<min_or_alt(lemma_list[lemma_list.word==x]["index"],1000000) for x in df_original[token_name]] 
            #token2_replacement_filter=~df_original.token2.isin(less_common_words) | ((~df_original.token2.isin(very_common_words))&(df_original.token2.str.len()<=3))
            print(df_original[token_replacement_filter].context+df_original[token_replacement_filter][token_name])
            replacement_info=[get_example_pairs(x,temperature=0,category="",stop=[" ",",",".",":"],max_total_tokens=20,n=1)
                                    for x in df_original[token_replacement_filter].context+df_original[token_replacement_filter][token_name]]
            token_replacements=[x["full_word"].iloc[0] for x in replacement_info]
            updated_probabilities=[x[prob_name].product() for x in replacement_info]
            original_tokens=df_original[token_replacement_filter][token_name]
            new_token=original_tokens+token_replacements
            original_probabilities=df_original[token_replacement_filter][prob_name]
            new_probability=original_probabilities*updated_probabilities
            new_token[~new_token.isin(less_common_words)]=original_tokens
            new_probability[~new_token.isin(less_common_words)]=original_probabilities
            df_original.loc[token_replacement_filter,token_name]=new_token
            df_original.loc[token_replacement_filter,prob_name]=new_probability
            print(f"{list(original_tokens)} -> {list(new_token)}")
            df_original=df_original[df_original[token_name].isin(less_common_words)]
        df_original["score"]=get_score(df_original["prob1"],df_original["prob2"],df_original["context_length"])
        return(df_original)
    df_filtered=df_filtered[~df_filtered['context'].apply(has_direct_repeated_words)]
    #Exclude examples that are part way through a word
    df_filtered=df_filtered[df_filtered['context'].str.endswith(' ')]
    df_filtered=df_filtered.drop_duplicates(subset=['id'])
    df_filtered=df_filtered[df_filtered["score"]>=0.7]
    #df_filtered=df_filtered[(df_filtered["prob2"]-df_filtered["prob3"])>0.05]
    df_filtered=df_filtered.reset_index()
    if len(df_filtered)>0:
        df_filtered["pair_keys"] = df_filtered.apply(lambda row: tuple(sorted([row['token1'], row['token2']])), axis=1)
        df_filtered=df_filtered[~df_filtered["pair_keys"].duplicated()]
        df_filtered=df_filtered[~df_filtered["context"].str.contains('|'.join(excl_list ),case=False)]
        df_filtered=df_filtered[~df_filtered["token1"].str.contains('|'.join(excl_list ),case=False)]
        df_filtered=df_filtered[~df_filtered["token2"].str.contains('|'.join(excl_list ),case=False)]
        df_filtered=update_table_with_longer_words(df_filtered)
        df_filtered=df_filtered[[word_to_lemma_map.get(x)!=word_to_lemma_map.get(y) for x,y in zip(df_filtered.token1,df_filtered.token2)]]
    
    else:
        df_filtered["pair_keys"]=""
    df_filtered["classification"]="potential"
    df_filtered["token1_stripped"]=df_filtered["token1"].str.strip()
    df_filtered["token2_stripped"]=df_filtered["token2"].str.strip()
    df_filtered=df_filtered[df_filtered["score"]>=0.7]
    return(df_filtered)
#%%

#%%
#word_list_4=["The sun over the", "In the quiet house", "Beyond the distant mountains","She heard the sound", "He looked at the", "Under the dark clouds","They walked toward the", "A sudden gust hit", "The stars above the","The forest near the", "As the storm approached", "In the early light","He had always seen", "She felt the presence", "The door to the","Through the mist, a", "Over the river, the", "His heart felt the","A shadow crossed the", "She could smell the", "He thought about the","They stood by the", "The wind carried the", "In the distance, the","She whispered to the", "The waves hit the", "He felt the ground","Underneath the tree, a", "As the rain started", "A flicker in the","The clock on the", "In the corner, the", "She couldn’t find the","He reached toward the", "The ground shook under", "As they neared the","The night over the", "His breath slowed as", "A light in the","She watched the sky", "The air held the", "They exchanged the look","He couldn’t forget the", "In the shadows, the", "Her eyes followed the","The silence in the", "The door to the", "In the distance, the","She turned toward the", "He picked up the", "A cold wind swept","They waited for the", "She felt the heat", "The sun set over","A soft glow lit", "He tried to recall", "She reached for the","As the night darkened", "The fire warmed the", "He closed the door","A flash in the", "The tension filled the", "He stood at the","Her heartbeat matched the", "The moonlight touched the", "The road led to","She knelt by the", "He was certain the", "They had reached the","She sighed with the", "His hands held the", "A strange echo filled","She hesitated before the", "He could not see", "The water lapped at","They whispered to the", "Her voice carried through", "The fog obscured the","A chill passed through", "He had waited by", "She saw the outline","The sky over the", "The room held the", "He stood before the","She couldn’t resist the", "A soft breeze stirred", "His footsteps led to","The door revealed the", "He heard the creak", "She felt the shift","The forest hid the", "A loud noise came", "She looked at the","He could hear the", "The flames reached the", "She caught the glimmer","A heavy fog enveloped", "He turned toward the", "The city lights blinked","The air carried the", "She reached for the", "In the blink, the","A distant sound echoed", "The trees blocked the", "He leaned against the"]
#df_word_list_4=pd.concat([get_example_pairs(x,temperature=0,category="word_list_4",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in word_list_4])
#df_word_list_4f
sentence_generation4_7=["The first glimpse of", "Every piece of the", "On the edge of", "She couldn’t ignore the", "As the sun set over", "With a heavy sigh, the", "It was the sound of", "In the middle of", "Without hesitation, the", "The distant sound of", "At the edge of", "He had always admired the", "Somewhere deep in the", "Underneath the blanket of", "Through the mist of", "Just as the door", "Despite the weight of", "At the break of", "By the end of the", "Lost in the maze of", "From the top of the", "In the distance, the", "As the door opened, the", "No one could deny the", "With trembling hands, the", "For as long as the", "Somewhere in the shadow of", "Beneath the towering", "She felt a sudden", "With the wind howling, the", "From behind the veil of", "He stood before the", "Without a second thought, the", "Amidst the chaos, the", "Every single one of the", "As the waves approached the", "With each passing moment, the", "The scent of fresh", "On the far edge of", "Through the open door, the", "The distant hum of the", "Somewhere between the pages of", "Over the rolling hills of", "With the moon above the", "In the fading light of the", "At the very last", "The chill in the", "As the fire blazed in the", "She whispered softly to the", "He could hardly see the", "In the silence of the", "Amidst the falling", "With her heart racing, the", "The echo of his", "On the other side of the", "In a quiet corner of the", "As the shadows danced across the", "With the sun low in the", "No one knew the weight of the", "By the time the", "With a sense of the", "He looked out toward the", "Underneath the bright light of the", "On the surface of the", "With each step toward the", "Somewhere in the distance, the", "In the blink of an", "Over the bridge, the", "As the music played in the", "At the sight of the", "By the light of the", "It was only when the", "As the clock chimed, the", "With a single look at the", "Lost in the thought of the", "She turned her gaze to the", "By the flickering light of the", "He stepped into the", "In the stillness of the", "She never imagined the", "At the sound of the", "With the first sign of the", "Through the narrow alleyway, the", "She had always feared the", "As the clouds covered the", "He found himself in the", "Just beyond the reach of the", "The wind whispered through the", "With the rain drenching the", "At the heart of the", "She reached out toward the", "In a world full of the", "As the sun climbed above the", "No one could understand the", "At the very top of the", "With a sigh, the", "She slowly closed her", "Through the crowded market, the", "Underneath the weight of the", "On the shores of the", "With the city in the", "The sound of laughter echoed in the", "Amidst the ruins of the", "With one last look at the", "At the thought of the", "In the gentle breeze of the", "By the edge of the", "Somewhere in the sprawling", "With the first touch of the","The first time a","Every one of the","On the edge of a","In the middle of the","Somewhere between the and","As the sun set on","The shadows cast by a","Through the doorway of the","In the distance, a","Along the winding path to","By the light of a","At the far end of","Under the cover of the","Beyond the reach of a","Through the haze of the","On the surface of the","With the sound of a","Within the confines of a","At the heart of the","Near the edge of a","Across the field of a","At the beginning of the","In the shadow of a","Above the curve of the","Behind the wall of a","Across the horizon, the","At the top of the","Inside the cavern of a","With the scent of a","Along the edge of the","On the brink of a","Under the shadow of a","Amidst the chaos of a","At the base of the","Underneath the weight of a","Beside the entrance of a","With the echo of a","On the cusp of a","Inside the warmth of a","At the bottom of a","With the whisper of a","At the edge of the","Across the bridge of a","Between the folds of the","On the floor of the","In the grip of a","Under the roof of a","At the threshold of a","Before the entrance to the","Near the top of the","Under the light of the","Within the depth of a","On the side of a","In the middle of a","Above the sound of the","Through the window of the","With the sight of a","Underneath the surface of the","Near the center of a","In the heart of a","With the arrival of a","On the outskirts of a","Before the sound of a","Within the reach of a","Inside the boundary of a","Underneath the skin of a","Behind the veil of a","At the intersection of the","In the warmth of a","Above the waves of a","Between the walls of the","On the threshold of a","At the corner of a","In the wake of a","Underneath the glow of the","By the side of a","In the shadow of the","On the trail of a","Near the foot of the","At the crossroads of a","Inside the boundaries of a","Beyond the edge of a","Through the fog of a","Under the blanket of a","With the dawn of a","In the silence of a","Above the clouds of a","Through the veil of a","By the river of a","At the crossroads of the","With the scent of the","On the verge of a","Behind the shadow of a","In the glow of a","Across the sands of a","On the winds of a","Under the surface of a","At the peak of a","In the quiet of a","Through the branches of a","Beyond the horizon of a","At the crest of a","On the steps of a","In the shelter of a","Under the influence of a","At the mouth of a","With the light of a","Through the gates of the","In the ruins of a","At the foot of a","By the shore of a","In the calm of a","On the edge of the","Underneath the surface of a","At the top of a","On the plains of a","Through the opening of a","Near the shore of the","With the touch of a","On the banks of a","At the rise of a","Beyond the gates of a","On the other side of","At the bottom of the","Under the watch of a","Within the walls of the","Near the horizon of a","On the tip of a","With the glow of a","In the air of a","At the edge of a","Near the opening of a","On the path of a","Through the ruins of the","Behind the doors of a","By the entrance to a","At the gates of a","Under the weight of the","Beyond the doors of a","On the floor of a","Beneath the looming shadow of","A single glance towards the","Resting quietly by the","With a sudden jolt, the","Faint whispers echoed from the","At a glance, the","No one could see the","If only the could have","Strangely, the silence of the","Hidden among the leaves, a","With each passing moment, the","A gentle breeze stirred the","Suddenly, out of nowhere, a","Nobody expected the arrival of","Underneath a sky full of","The murmur of voices near the","Without warning, the","A strange stillness fell over the","Drawn by the sound of a","Through a crack in the","Far off in the distance, the","The world seemed to hold its","Once upon a time, a","Caught in the flickering light of a","Before anyone noticed, the","As if in response, the","With every step closer to the","The warmth of a lingering","Hanging in the air, the","A sudden burst from the","Behind closed doors, the","Long forgotten by the","In the wake of the","Glancing over their shoulder, the","Time itself seemed to slow as the","An unexpected gust rattled the","Through the heavy mist, a","In the quiet moments after the","As the storm clouds gathered, the","Pressed against the cold stone of the","Distant cries could be heard from the","In the midst of chaos, the","Fleeting moments of clarity came to the","Under the weight of a","Before the sun could rise, a","With great care, the","For a brief moment, the","Just beyond the horizon, the","Amidst the laughter, the","Lingering in the silence, the","In the reflection of the","Despite the warning, the","Between the flickers of light, a","Hidden in plain sight, the","As the dust settled, the","With no time to waste, a","Hovering just above the","Without a second thought, the","After all had fallen quiet, the","In the heart of the storm, the","Standing alone on the","Before the night consumed the","The faint glimmer of a distant","Caught off guard by the","Waiting patiently for the","Through the shattered glass, the","In the blink of an eye, the","The memory of a once-forgotten","Barely visible through the thick fog, the","Against the backdrop of a","Every corner of the room seemed to hide a","Without a sound, the","Far from civilization, the","Glimpses of the past lingered in the","Once the dust cleared, the","If only the had seen it coming","In a world filled with","Before the light could fade, the","For reasons unknown, the","After the storm had passed, the","Buried deep within the earth, a","As the hours dragged on, the","Beside the old oak tree, a","No one could have guessed the","Behind the veil of secrecy, the","On the far side of the","A brief flicker in the darkness revealed a","Long before the dawn of the","Beneath layers of time, the","In the quiet of the night, a","From the depths of the","Despite the odds, the","At the breaking of the","Within the hidden corners of the","In the aftermath of the","Where the mountains met the","Before the world could react, a","While the rest of the world slept, a","Through the gaps in the","A sudden realization dawned on the","Without a clear path forward, the","As the day began to fade, the","In the final moments before the"]
formulaic_noun_drop=["At the heart of the","By the glow of the","With a smile on her","Under the weight of the","Beyond the reach of the","After the fall of the","Before the dawn of the","Over the crest of the","Through the depths of the","Within the walls of the","Near the edge of the","Beside the warmth of the","Amid the chaos of the","Behind the shadows of the","Under the gaze of the","Upon the wings of the","In the stillness of the","Across the plains of the","Outside the gates of the","Between the silence of the","Over the bridge of the","Beside the ruins of the","Beyond the veil of the","Across the fields of the","Near the shores of the","With the scent of the","Under the branches of the","By the side of the","Before the rise of the","Beyond the borders of the","Over the sound of the","Under the cover of the","Beside the edge of the","Upon the peak of the","Through the heart of the","In the shadow of the","At the height of the","Under the shelter of the","Across the waves of the","Beside the path of the","With the grace of the","After the call of the","Beyond the sight of the","Before the start of the","Near the end of the","Over the crest of the","Amid the chaos of the","Between the whispers of the","Beside the light of the","Under the warmth of the","With the strength of the","Beyond the walls of the","Over the edge of the","Beside the stillness of the","In the silence of the","Under the spell of the","By the shores of the","Before the fall of the","Upon the wings of the","With the wisdom of the","Across the horizon of the","Beyond the gates of the","Near the heart of the","Under the watch of the","Through the light of the","By the side of the","Across the expanse of the","Over the depth of the","With the roar of the","By the glow of the","Through the darkness of the","Across the fields of the","With the echo of the","Beyond the reach of the","Upon the summit of the","With the scent of the","Before the break of the","Beside the flow of the","Under the cover of the","With the song of the","Beyond the flames of the","Under the gaze of the","Through the mist of the","Across the surface of the","At the edge of the","Upon the crest of the","With the glow of the","Beside the calm of the","By the light of the","Beyond the reach of the","Near the end of the","Upon the edge of the","At the height of the","With the crackle of the","Under the shadow of the","Beside the warmth of the","Upon the back of the","Before the dawn of the","Across the surface of the","Beyond the walls of the","Under the blanket of the","By the flicker of the","With the rumble of the","Under the branches of the","Through the doorway of the","Beyond the light of the","With the howl of the","Upon the back of the","Beside the sound of the","Over the edge of the","Upon the face of the","By the shore of the","Within the reach of the","Before the rise of the","By the hum of the","Under the arch of the","Beside the sound of the","Beyond the boundaries of the","By the gate of the","Through the windows of the","With the whisper of the","Upon the wings of the","Near the banks of the","Upon the plains of the","Over the lands of the","Before the return of the","Beside the song of the","By the glow of the","Within the walls of the","Over the sound of the","Upon the silence of the","With the chirp of the","Beyond the peak of the","By the scent of the","Upon the trail of the","Across the sea of the","With the chill of the","By the foot of the","Under the glow of the","At the crest of the","Within the boundaries of the","Under the flight of the","Over the hills of the","Upon the branches of the","Through the curtain of the","Before the storm of the","By the walls of the","Under the power of the","By the light of the","Within the folds of the","Over the edge of the","Before the fall of the","Beside the stillness of the","Under the branches of the","With the warmth of the","Over the cliffs of the","Under the watch of the","Upon the wings of the","Beside the shadow of the","Across the expanse of the","Beyond the reach of the","By the hum of the","Through the doors of the","Upon the waves of the","By the side of the","Upon the back of the","Beside the beauty of the","Beyond the grasp of the","Upon the peak of the","At the gates of the","With the shimmer of the","Before the break of the","Upon the leaves of the","By the strength of the","With the chill of the","Upon the wings of the","Under the weight of the","With the buzz of the","Upon the edge of the","Before the turn of the","Under the branches of the","With the courage of the","By the sound of the","Upon the crest of the","Across the sands of the","With the glow of the","Beyond the point of the","Over the horizon of the","With the strength of the","Upon the shores of the","Beside the ruins of the","By the crackle of the","Under the glow of the","Upon the waves of the","Across the streets of the","With the scent of the","By the branches of the","Through the roar of the","Beside the ruins of the","Upon the fields of the","Under the blaze of the","Across the tracks of the","With the call of the","Under the wings of the","Upon the flow of the","With the glow of the","Across the width of the","By the fire of the","With the power of the","Across the breadth of the","By the wings of the","Under the shine of the","Before the turn of the"]
variety_noun_drop=["A whisper carried through the","Every word lingered in the","She paused, watching the","Nothing seemed out of","The silence held a strange","His eyes fixed on the","Clouds gathered above the","The streets echoed with","A secret lay beneath the","He waited by the old","Each step echoed in the","Her gaze fell upon the","The world felt distant and","Shadows danced along the","Rain tapped softly on the","Voices faded into the","Memories lingered in the cold","A light flickered in the","She stood alone in the","Time slipped through her","A gentle breeze rustled the","His thoughts wandered back","The forest hummed with","A chill crept through the","The air was thick with","Her heart raced with","Stars blinked in the vast","The path wound through the","Birds chirped in the early","A single flower bloomed in the","Footsteps echoed down the","Silence settled over the","A heavy mist blanketed the","The door creaked in the","Wind howled across the open","The fire crackled in the","He waited by the old","Her laughter filled the","An eerie stillness took","She clutched the note","Dew glistened on the","He glanced at the clock","The book lay open on the","Light streamed through the","Footsteps approached from the","A distant melody drifted through the","The room felt colder than","He stood frozen in the","Her hand trembled as she reached","Nothing moved in the still","The stars shimmered above the","A single tear slid down her","His shadow stretched across the","The echo of footsteps filled the","Warm sunlight filtered through the","She exhaled slowly, watching the","The wind whispered secrets in his","A sharp pain shot through his","The walls seemed to close","Her voice broke the","The air smelled of fresh","A soft hum filled the","He felt the weight of his","Dark clouds loomed on the","A flicker of hope crossed his","The old house creaked in","Her footsteps quickened as she","He wiped the sweat from his","Time seemed to slow around","The sky burned with the setting","Cold water splashed against the","The sound of laughter echoed","She caught a glimpse of his","Fog rolled in from the","A quiet calm settled over the","His hands shook with","Her heart pounded in her","The forest grew darker with each","A strange light flickered in the","He glanced nervously at his","Snowflakes danced in the winter","Her eyes searched the empty","The fire roared in the","Heavy footsteps echoed behind","The garden bloomed with","His breath came in short","The road stretched on for","The moon hung low in the","Her fingers grazed the cool","A heavy fog blanketed the","Leaves crunched beneath her","He watched the river flow","A sudden gust blew through the","Thunder rumbled in the","Her pulse quickened with every","Snow blanketed the sleeping","His thoughts drifted far","The stars twinkled like distant","Raindrops fell gently on the","She felt the warmth of his","The wind carried his words","Her reflection stared back at","Shadows stretched across the","His fingers brushed the cold","The city lights flickered in the","A soft breeze ruffled her","The fire crackled, casting long","He stood at the edge of the","The sky darkened as the storm","A gentle rain began to","Her hand lingered on the","He felt a chill run down his","The horizon glowed with the rising","Silence enveloped the abandoned","The river rushed past, swift and","A faint smile played on her","His eyes scanned the","Snow fell silently in the","The scent of pine filled the","Her laughter echoed through the","The streets were eerily","A sense of dread filled the","She stood tall, facing the","He watched the clouds gather","The waves crashed against the","Her fingers traced the edge of the","The candle flickered, casting soft","His footsteps left prints in the","A thick fog obscured the","The car engine roared to","Her hand gripped the steering","His heart ached with","She felt the tension in the","The fire burned brightly in the","The rain poured down in","He felt the ground shift beneath","Her eyes narrowed as she","The wind howled through the","A bird sang from a distant","The door swung open with a","She glanced nervously at her","His footsteps echoed in the","The sun dipped below the","Her fingers grazed the piano","He could hear the distant","The room was bathed in","She stood silently in the","His breath fogged up the","The smell of coffee filled the","The forest floor was soft","Her voice cracked as she","The stars seemed closer than","He reached out, but grasped only","The sky glowed with the coming","A soft laugh escaped her","The room felt warmer","His mind raced with","She wiped away a stray","The rain pattered softly on the","The waves lapped gently at the","He felt a sense of","The fire's warmth embraced","The streetlights flickered","Her heart fluttered in her","He could smell the fresh bread","A distant bell chimed the","Her gaze lingered on the","He closed his eyes, breathing","She traced the patterns on the","The road stretched out before","Her pulse quickened with every","Shadows moved in the","The silence weighed heavily on","He stood at the","The rain began to","She felt the tension leave her","The clouds parted, revealing the","A soft tune played on the","His fingers tapped nervously on the","She looked back one last","The forest whispered in the","The night was still and","He sighed, staring at the empty","The scent of lavender filled the","She watched the rain trail down the","He shivered, pulling his coat","The sun broke through the","She tucked a strand of hair behind her","The boat rocked gently on the","A single light flickered in the","His fingers hovered over the","Her heels clicked on the marble","The leaves rustled softly in the","He glanced over his","The wind tugged at his","She knelt by the","A shadow moved in the","The sun cast long shadows across the","Her breath came in shallow","He leaned against the","Clouds drifted lazily across the","The warmth of the sun touched her","She glanced out the window, deep in","His hand rested on the cold","The lights of the city sparkled","Her voice was barely a","His hand brushed against","Snow crunched beneath their","She ran her fingers through the soft","The scent of fresh rain filled the","A bird chirped softly in the","His voice broke through the","She could feel the cool breeze on her","The horizon glowed with the setting","Footsteps shuffled on the dusty","The house was quiet, too","He tilted his head, listening","Her eyes closed as she listened to the","The sound of waves filled the","She hugged her coat closer against the","His thoughts swirled like a","The street was bathed in soft","Her fingers traced the rim of the","The city buzzed with","He leaned forward, resting his chin on his","The air was heavy with","She turned the key, holding her","A distant train whistle echoed through the","The stars seemed brighter than","His eyes were fixed on the","She reached out to touch the cool","Leaves swirled in the autumn","He felt the weight of the world on his","The night sky stretched endlessly","Her fingers brushed against the cold","A bird soared high above the","She stared at the empty","His voice was low, barely","The room smelled of fresh","She closed her eyes, taking a deep","The sun warmed her face as she","His hand hovered over the","The night air was crisp and","She felt the warmth of the","A soft glow illuminated the","He stood quietly in the","The breeze carried the scent of","Her fingers tapped nervously on the","Rain drummed on the roof","His gaze followed the fading","She walked slowly through the empty","A soft hum filled the","He could hear the clock","The silence between them","Her hand rested lightly on his","The scent of smoke lingered in the","His heart sank at the","She felt the ground shift beneath","The stars blinked through the","His footsteps echoed in the empty","She could taste the salt in the","The moon cast a pale","His voice was steady, but","She watched the shadows grow","A chill crept into the","Her pulse raced as she","The trees swayed gently in the","His breath came in short","She could feel his eyes on","The sun dipped below the","Her mind raced with","The candle flickered in the","His fingers brushed her","The waves lapped softly at the","She stood alone on the","The room was filled with soft","Her eyes widened with","The air was thick with","He watched the flames dance in the","Her hands trembled as she reached","The sound of water echoed in the","She felt the cold wind on her","The sky was painted with shades of","His voice cut through the","A soft knock broke the","Her hand tightened around the","The sun set in a blaze of","His mind wandered to distant","She leaned against the old oak","The wind tugged at her","His fingers curled around the edge of the","A single raindrop slid down the","The sky darkened as the storm","She could feel the tension in the","The city lights twinkled","His laughter echoed in the small","She watched the clock tick","The air felt thick and","Her thoughts drifted to the","The street was quiet, almost too","His hands clenched into","She stood at the edge of the","The cold seeped into her","A low murmur spread through the","His eyes darted around the","She could hear the faint sound of","Rain splashed against the","The leaves rustled in the soft","Her heart ached with","The clouds drifted lazily across the","He stood motionless in the","She felt the warmth of the fire on her","The silence was broken by a distant","His eyes were full of","She clutched the letter tightly in her","A soft breeze rustled the","He gazed out at the open","Her mind was filled with unanswered","The stars twinkled in the clear night","His footsteps were heavy and","She took a deep breath, steadying","The distant sound of thunder","His heart beat faster with every","She leaned forward, listening","The fog rolled in, thick and","His eyes lingered on the","She felt the weight of his","The streetlights cast long","A sense of dread settled over","She stared blankly at the","His voice was barely more than a","The room grew colder by the","Her fingers tightened around the","The wind howled outside the","His breath fogged up the","She watched the flames flicker and","A single tear fell from his","The city was alive with","Her heart pounded in her","The sun peeked through the","He felt a chill run down his","The door creaked open","Her smile lit up the","His hands shook as he","She watched the raindrops race down the"]
full_clauses_drop=["The wind howled through the","She stared at the old","A flicker of hope ignited","Thunder rumbled ominously in the","The candle flickered as shadows","He felt a chill run","A distant bell tolled midnight","She wrapped her shawl tightly","The clock ticked away the","Stars twinkled in the night","He stumbled upon an ancient","The rain drummed against the","A soft breeze whispered through","He held his breath in","Her laughter echoed through the","Shadows merged into one dark","The air smelled of fresh","A door creaked open","They shared a secret glance","The streets were empty and","She clutched the letter in","A cat purred softly on the","The fog rolled in like","He traced her name in the","An old man sat on the","A scream pierced the stillness","The fire crackled in the","She paused to admire the","His heart raced with every","A shadow lurked just out of","The taste of salt lingered on","A distant melody played in the","She walked barefoot on the","The old house groaned with","Clouds gathered, darkening the afternoon","He scribbled furiously in his","A smile tugged at her","The waves crashed against the","A sudden gust of wind","She felt a warmth envelop","The night was filled with","He gazed out at the","A chill settled in her","She opened the book to chapter","The scent of jasmine filled the","He glanced back over his","The room was filled with","She brushed her hair away","The ground trembled beneath their","A flicker of doubt crossed his","She picked up the fallen","The shadows lengthened as dusk","A single tear rolled down her","The air crackled with electric","He stared into the fireplace,","A light flickered in the","The leaves rustled in the gentle","He whispered a prayer for","She closed her eyes in","The world outside felt distant and","He could feel the weight of","A sudden noise broke the","She traced her fingers over the","The candlelight illuminated her features","He felt the warmth of her","The walls echoed with forgotten","She wrapped her arms around","The sun peeked through the","A distant train whistle called","She breathed in the crisp morning","The garden was alive with","He picked up the forgotten","The clock chimed the hour","A feeling of unease settled","She glanced at the empty","The stars seemed to shimmer","He took a step into","A flash of lightning illuminated the","The whispers grew louder in her","A gentle rain began to","The old tree stood tall and","She recalled the promise he","A sense of adventure filled the","He reached for the doorknob","The moon hung low in the","She felt a sudden surge of","The smell of fresh bread wafted","He couldn’t shake the feeling of","The shadows danced in the flickering","She turned the page with","A sense of nostalgia washed over","The sound of footsteps approached","He savored the moment of","A rustle in the bushes startled","The rain painted the world","She closed the window against the","The ocean waves sang a","He watched the sun rise","A gust of wind swept","The scent of pine filled her","He traced the outline of her","The laughter of children filled the","She felt the cool grass","A flickering light caught his","The pages turned with a soft","He lingered in the doorway,","The streets glistened after the","She clutched the gift tightly to her","A voice called out in the","The firelight danced across their","He took a deep breath,","The night sky sparkled with","She found solace in the","The sound of rustling leaves surrounded","A fleeting moment of clarity","The old barn creaked under the","She stared at the horizon in","A faint echo lingered in the","He felt an inexplicable connection to","The sound of waves brought","She dared to dream of","A gentle hand squeezed her","The wind whispered secrets through","He felt the heat of the","A single raindrop fell on her","The garden was a riot of","She took a leap of","A shadow crossed his mind","The distant mountains loomed","He closed his eyes, lost in","A feeling of calm washed over","The sun shone brightly on the","He heard a soft sigh behind","She admired the intricate details","A wildflower caught her eye","The room fell silent as she","He watched the clouds drift","The taste of victory lingered","She held the key tightly in","The shadows merged into one dark","A flicker of recognition flashed","He stepped into the unknown","The world felt suspended in","She listened to the rustling","A new chapter awaited them","The night enveloped them in","She felt the warmth of the","He held her gaze,","The clock ticked louder in the","A distant memory came flooding","She felt the weight of her","The walls seemed to close in","He reached for the last","The shadows stretched across the","She stepped into the light with","A smile broke across his","The sound of laughter echoed","He glanced around, searching for","A sudden burst of color","She felt alive in that","The world outside was a","He took a moment to","The sound of rain was","She watched the sunrise with","A sudden realization dawned upon","The echoes of the past","He paused to take it all","A warmth spread through the","The stars whispered ancient secrets","She turned her back on the","A new adventure awaited beyond the","He marveled at the beauty of","The taste of chocolate melted in her","She heard the call of the","The door swung open with a","He stepped forward, heart pounding","The wind carried her words","She felt a rush of","The horizon promised a new","He felt the weight of the","A flickering flame illuminated the","She closed her eyes, dreaming of","The sound of waves soothed her","He glanced at her, lost in","The scent of fresh coffee filled the","She felt free for the first","The sun dipped below the","He stood alone, contemplating his","A glimmer of hope shone","She reached out, touching his arm","The night wrapped around them","A moment of silence enveloped","He couldn’t shake the feeling of","The firelight danced across her","She smiled, feeling a sense of","The shadows played tricks on her","A soft sigh escaped his","She knew the journey was just","The beauty of the moment overwhelmed","He took a step into the"]
clause_generation_gpt4o=["The sun dipped below the","A shadow crept across the","She whispered softly to her","In the quiet of the","He took a deep breath and","The wind rustled through the","As night fell, the stars","They walked hand in hand towards the","A chill ran down his","With a flick of her","The old house creaked as","She gazed out at the","He could hear the distant","In the garden, flowers bloomed with","The clock struck midnight, and","He opened the letter with","The river flowed gently past the","With a heavy heart, she","The crowd cheered as the","She closed her eyes and","He felt a surge of","The fire crackled in the","A voice called out from","As the sun rose over the","They shared a knowing glance as","The door swung open with a","She found a hidden path leading to the","A storm was brewing on the","He reached for the last","The city lights twinkled like","With each step, she felt more","The music played softly in the","He couldn't shake the feeling of","She walked through the forest, surrounded by","The air was thick with","He gazed at the painting, lost in","As the train pulled away from the","The child laughed as they chased the","A tear rolled down her","They shared stories around the","The smell of fresh bread filled the","He noticed something unusual about the","As dusk settled, the sky turned","She turned the page with a","In the distance, thunder rumbled like","He felt the weight of the","The moon hung low in the","With every heartbeat, she felt the","The scent of roses lingered in the","He knelt down to tie his","The children played joyfully in the","She took a moment to gather her","As they danced, the world around them","He watched as the clouds gathered in the","The stars shone brightly over the","With a sudden rush of courage, she","The old man smiled at the","She heard a rustling in the","As he entered the room, silence","The leaves fell gently to the","He glanced at his watch, feeling the","With a sigh, she set down her","The sound of waves crashing echoed in the","In the morning light, everything felt","They knew this moment would change their","He took a step back, unsure of","The fireplace crackled with warmth as","A mysterious figure appeared at the","She could see the outline of a","With a twist of fate, their paths","The taste of salt lingered on his","As the day turned to night, they","He ran his fingers over the","A bright light pierced through the","In the garden, shadows danced beneath the","She held her breath as the","A secret whispered through the old","With a leap of faith, she","The night air was filled with","He traced the map with his","With every word, the tension in the","As they reached the summit, a sense of","The thunder rolled like a","She stepped into the unknown, her heart racing","The waves crashed against the rocky","He couldn’t believe his eyes when","As the sun set, the sky turned","She felt an inexplicable pull towards the","The ground trembled beneath their","In the quiet corners of her","He discovered an old book filled with","With each passing moment, hope began to","The shadows grew longer as the","As he turned the corner, he saw","The wind howled through the empty","With a deep sigh, she opened the","The treasure lay buried beneath the","In the corner of the room, a","He couldn't ignore the feeling that","The streetlights flickered, casting eerie","She gazed longingly at the","The taste of adventure lingered in his","With each heartbeat, he felt the","The house was filled with echoes of","In the twilight, everything felt surreal and","He clutched the photograph tightly to his","As the door creaked open, a","A glimmer of hope shone through the","She found solace in the sound of","With every heartbeat, the world around them","The air crackled with anticipation as","He could feel the warmth of the","The past haunted him like a","In the silence, she heard a","He stared into the distance, searching for","With trembling hands, she opened the","The lighthouse stood tall against the","She wrapped her arms around herself, feeling","The echo of laughter filled the","He stepped closer, curiosity overcoming his","In the depths of the forest, darkness","She felt the weight of the world on","A flicker of light caught his","The landscape stretched endlessly before her","With each brushstroke, she brought her","He felt a chill in the","The festival lights twinkled like stars in","She paused, listening to the rhythm of","In the depths of winter, everything felt","He took a deep breath and","A promise lingered in the air as","She gazed at the horizon, searching for","The past and future collided in a","He closed his eyes, feeling the","With a spark of inspiration, he","The garden was alive with the sounds of","She couldn't shake the feeling that","In the depths of his heart, he","He heard the distant sound of","The sun broke through the clouds, illuminating","She reached for the stars, hoping to","He traced the line of her","As shadows lengthened, the air turned","A secret lay hidden beneath the","She turned to face him, determination in","In the silence, the world seemed to","He felt the magic in the","With a soft sigh, she remembered the","The clock chimed, signaling the end of","She knew this moment was too precious to","He stepped into the light, embracing the","As dawn approached, the world awakened with","The path ahead was uncertain, but","With a gentle touch, he brushed away","The laughter of children echoed in the","She clung to the hope that","In the stillness, dreams began to","He looked up at the night sky, searching for","The mountains loomed majestically in the","With each heartbeat, he felt more alive than","She picked up the pen, ready to","The horizon glowed with the promise of","He took a step forward, fueled by","A flicker of doubt crossed his","She felt the pull of destiny in the","The storm raged outside, but inside, they felt","He watched the raindrops race down the","As the sun set, shadows danced across the","With a smile, she embraced the uncertainty of","The old tree stood proudly in the","He stumbled upon an ancient map leading to","In the garden, secrets waited to be","The train whistle echoed in the quiet","With hope in her heart, she took the","He stared at the horizon, contemplating his","As the fire burned low, stories unfolded in the","The city buzzed with life as night","He gathered his courage, ready to face","The melody lingered in her mind, haunting and","In the distance, a bell tolled ominously in the","She felt a sense of belonging in the","The book opened to a page filled with","He followed the sound of laughter through the","As the night deepened, the shadows began to","She cherished the memories of a time long","With a heart full of hope, she stepped into the","The walls held whispers of forgotten","He peered into the darkness, seeking a","The scent of pine filled the crisp","In the distance, a flicker of light promised","With every breath, she felt more connected to","He realized that love was worth the","The garden was alive with colors and","As they spoke, the world faded into the","With each word, the tension began to","He listened closely, hoping to catch a","The fog rolled in, shrouding the world in","She followed the path, trusting it would lead her to","The sun dipped below the horizon, painting the","With a last glance back, he stepped into the"]
all_words=sentence_generation4_7+formulaic_noun_drop+variety_noun_drop+full_clauses_drop+clause_generation_gpt4o
print("a")
a=[get_example_pairs(x,temperature=0,category="sentence_generation4_7",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in sentence_generation4_7]
print("b")
b=[get_example_pairs(x,temperature=0,category="formulaic_noun_drop",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in formulaic_noun_drop]
print("c")
c=[get_example_pairs(x,temperature=0,category="variety_noun_drop",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in variety_noun_drop]
print("d")
d=[get_example_pairs(x,temperature=0,category="full_clauses_drop_gpt4o-mini",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in full_clauses_drop]
print("e")
e=[get_example_pairs(x,temperature=0,category="clause_generation_gpt4o-mini",stop=[" ",",",".",":"],max_total_tokens=20,n=1) for x in clause_generation_gpt4o]
print("filtering...")
df_word_list=pd.concat(a+b+c+d+e)

df_word_list=pd.read_csv("outputs/gpt4_generation_raw.csv")

agg_dict = {}
agg_dict["token1"] = lambda x: ''.join(x)
agg_dict["token2"] = 'first'
agg_dict['prob1'] = 'prod'
agg_dict['cum_prob2'] = 'prod'
agg_dict.update({col: 'first' for col in df_word_list.columns if col not in ["id", "token1","token2","prob1","cum_prob2"]})
duplicate_sentences=pd.Series(all_words)[pd.Series(all_words).duplicated()].unique()
duplicate_results=df_word_list[df_word_list.start_text.isin(duplicate_sentences)]
df_word_list=df_word_list[~df_word_list.start_text.isin(duplicate_sentences)]
result = df_word_list.groupby('id').agg(agg_dict).reset_index()

filtered_results=filter_results(result)
#filtered_results=update_table_with_longer_words(filtered_results)
filtered_results.to_csv("outputs/gpt4_generation.csv",index=False)
#df_word_list.to_csv("outputs/gpt4_generation_raw.csv",index=False)


#%%
potential_items=filter_results(df_word_list_4)
df_refined_list=[]
for row in (potential_items).itertuples():
    out=get_example_pairs(row.context,max_total_tokens=20,
    temperature=0,category=row.category,n=1,stop=[" ",",",".",":"])
    df_refined_list.append(out)
df_refined=filter_results(pd.concat(df_refined_list))
df_refined["classification"]="reviewed"
#%%
# for i in np.arange(0.0,0.8,.025):
#     print(f"{i} {len(df_filtered[df_filtered['score']>=i])}")

# df=pd.merge(df, x, how='left', on=["token1_stripped","token2_stripped"],suffixes=["",".new"])
# df["classification"]=df["classification.new"].fillna(df["classification"])
# df=df.drop("classification.new",axis=1)
#%%
x=pd.DataFrame(
[["corridor",	 "hallway", "reviewed"],
 ["picked",	 "began", "reviewed"],
 ["trends",	 "developments", "reviewed"],
 ["good",	 "warm", "reviewed"],
 ["seek",	 "find", "reviewed"],
 ["landscape",	 "world", "reviewed"],
 ["conversations",	 "events", "reviewed"],
 ["grace",	 "precision", "reviewed"],
 ["times",	 "eras", "reviewed"],
 ["soothing",	 "rhythmic", "reviewed"],
 ["backdrop",	 "atmosphere", "reviewed"],
 ["soft",	 "gentle", "reviewed"],
 ["air",	 "breeze", "reviewed"],
 ["dusk",	 "twilight", "reviewed"],
 ["faint",	 "soft", "reviewed"],
 ["leaves",	 "whispers", "reviewed"],
 ["filled",	 "illuminated", "reviewed"],
 ["leaves",	 "trees", "reviewed"],
 ["sky",	 "clouds", "reviewed"],
 ["wind",	 "autumn", "reviewed"],
 ["aspirations",	 "promises", "reviewed"],
 ["smile",	 "touch", "reviewed"],
 ["air",	 "room", "reviewed"],
 ["countless",	 "stars", "reviewed"],
 ["dim",	 "dark", "reviewed"],
 ["soft",	 "pale", "reviewed"],
 ["frost",	 "landscape", "reviewed"],
 ["purple",	 "pink", "reviewed"],
 ["vibrant",	 "deep", "reviewed"],
 ["cutting",	 "breaking", "reviewed"],
 ["trembling",	 "hovering", "reviewed"],
 ["bustling",	 "crowded", "reviewed"],
 ["leather",	 "spine", "reviewed"],
 ["overwhelmed",	 "worn", "reviewed"],
 ["canvas",	 "tapestry", "reviewed"],
 ["coat",	 "jacket", "reviewed"],
 ["territory",	 "wilderness", "reviewed"],
 ["businesses",	 "organizations", "reviewed"]],
 columns=["token1_stripped","token2_stripped","classification"]
)
#%%
import firebase_admin
from firebase_admin import credentials,db,firestore


cred = credentials.Certificate("dependencies\potato-jem-firebase-adminsdk-s2fjd-fe5af70ac8.json")
firebase_admin.initialize_app(cred)#, {
  #"databaseURL": "https://potato-jem-default-rtdb.asia-southeast1.firebasedatabase.app",
#})
db = firestore.client()
#%%
colnames=['token1_stripped','token2_stripped','prob1','prob2','classification','category','context','score']
manual_data=[
    ["air","breeze",0.67796,0.32036,"review","","the cool evening",get_score(0.67796,0.32036,4)],
    ["vibrant","deep",.8,.2,"review","","He loved the colour, it was a",get_score(.8,0.2,7)],
    ["ocean","water",0.75,0.24,"review","","Underneath the surface of the",get_score(.98,0.24,4)]
]

df_manual=pd.DataFrame(manual_data,columns=colnames)

for row in df_manual.itertuples():
    #print(row.token1.strip())
    data = {
        'target1':row.token1_stripped,
        'target2':row.token2_stripped,
        'prob1':row.prob1,
        'prob2':row.prob2,
        'classification': row.classification,
        'category':row.category,
        'answer': row.context,
        'difficulty': 0,
        'stale': False,
        'user_count': 0,
        'score':row.score
    }
    ref = db.collection(f"targets/PROD1/{row.classification}_items")
    ref.add(data)


# %%
input_text="shrouded in" #Mysyery/darkness
input_text="Plants grow on other planets, such as" #Venus/potato
input_text="In the morning, first I make my" #bed/cofee
input_text="The door was secured with a" #lock/key
input_text="Symbolising opening, the necklace had a tiny" #lock/key
n=3
t=3
response_main = client.chat.completions.create(
    model="gpt-4o-mini"#"gpt-3.5-turbo"
    ,messages=[
        {"role": "system",
         "content": "continue the text"},
        {
        "role": "assistant",
        "content": input_text#"finish this: "+
        }#,
        # {
        # "role": "user", 
        # "content": input_text
        # }
        ]
    ,max_tokens=t
    ,temperature=0
    ,logprobs=True
    ,top_logprobs=10
    #,stop=[".",","]
)
import numpy as np
import pandas as pd
tokens=[(z.token,z.logprob) for z in response_main.choices[0].logprobs.content[0].top_logprobs]
df=pd.DataFrame(tokens,columns=["token0","logprob"])
df["index"]=df["token0"]
df=df.set_index("index")

first_follow="".join([z.token for z in response_main.choices[0].logprobs.content][1:])
df=df[~df["token0"].str.lstrip().duplicated()]
df["token_r"]=None
df.loc[df.index[0],"token_r"]=first_follow
df
#%%
token_r = [client.chat.completions.create(
      model="gpt-3.5-turbo"
      ,messages=[{"role": "user", "content": input_text+k}]
      ,max_tokens=t-1
      ,temperature=0
      #,stop=[".",","]
      # ,stop=[" ",".",","]
  )
 for k in df.index[1:min(n,len(df.index))]
]

df.loc[df.index[1:min(n,len(df.index))],"token_r"]=[r.choices[0].message.content for r in token_r]

df["tokenAll"]=df["token0"]+df["token_r"]
df["nextWord"]=df["tokenAll"].str.extract(r'(\s*\S+)', expand=False)

#updated_tokens=[(input_text,tokens[i][0],token_r[i-1].choices[0].message.content if i>0 else first_follow,tokens[i][1]) for i in range(0,len(tokens[0:n]))]
df[~df["nextWord"].duplicated()]
#updated_tokens.insert(0,tuple([input_text,tokens[0][0],first_follow,tokens[0][1]]))
#updated_tokens
df
#%%
responseX = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": input_text}],
    max_tokens=5,
    #temperature=0,
    #top_p=1,
    #logprobs=True,  # Retrieve log probabilities for the top 5 tokens
    #top_logprobs=3
)


# Extract the logprobs from the response
# log_probs = response['choices'][0]['logprobs']
# tokens = log_probs['tokens']
# token_log_probs = log_probs['token_logprobs']

# print("Token predictions with probabilities:")
# for token, log_prob in zip(tokens, token_log_probs):
#     prob = math.exp(log_prob)  # Convert log probability to regular probability
#     print(f"Token: {token}, Probability: {prob:.4f}")
# %%
x=[-0.7379502,
-0.80859077,
-4.181693,
-4.5877066,
-4.7620707,
-5.392024,
-5.5443583,
-5.701539,
-5.736581,
-5.839223]

from itertools import accumulate

y=[math.exp(z) for z in x]
cum_sum = list(accumulate(y))

#%%

words_2_upper=["Curious minds",
"Bright lights",
"Gentle breeze",
"Sudden rain",
"Quiet footsteps",
"Golden rays",
"Soft whispers",
"Heavy clouds",
"Broken dreams",
"Silent streets",
"Quick glance",
"Warm hands",
"Cold nights",
"Dark shadows",
"Faint smiles",
"Deep waters",
"Tall trees",
"Hidden secrets",
"Empty rooms",
"Lonely hearts",
"Twinkling stars",
"Falling leaves",
"Flickering lights",
"Distant echoes",
"Restless nights",
"Forgotten memories",
"Brave souls",
"Shattered glass",
"Dancing flames",
"Gentle touch",
"Whispered promises",
"Velvet curtains",
"Ancient ruins",
"Mystic forest",
"Burning desire",
"Chilly winds",
"Rolling hills",
"Velvet skies",
"Snowy peaks",
"Rising sun",
"Frozen lakes",
"Lost time",
"Hidden paths",
"Distant voices",
"Mysterious figures",
"Barren lands",
"Bright moon",
"Old books",
"Quiet conversations",
"Tangled thoughts",
"Silent tears",
"Gentle waves",
"Midnight dreams",
"Rustling leaves",
"Fading light",
"Crashing waves",
"Twisted branches",
"Forgotten roads",
"Lonely figure",
"Raging storm",
"Golden sunset",
"Cool breeze",
"Midnight blue",
"Fiery eyes",
"Painted skies",
"Murmuring streams",
"Winding road",
"Broken clock",
"Silver lining",
"Dancing shadows",
"Shining stars",
"Warm glow",
"Rising smoke",
"Flickering flame",
"Trembling hands",
"Echoing footsteps",
"Twinkling lights",
"Cold dawn",
"Hazy memories",
"Blinding light",
"Silent night",
"Bright colors",
"Stormy weather",
"Gentle breeze",
"Worn pages",
"Spinning top",
"Fragrant flowers",
"Creeping fog",
"Whispered words",
"Soft raindrops",
"Endless horizon",
"Velvet touch",
"Lonely road",
"Crimson sky",
"Dark night",
"Rustling pages",
"Soothing music",
"Fading echo",
"Golden fields",
"Midnight hour"]

words_2_lower=[
"curious minds",
"bright lights",
"gentle breeze",
"sudden rain",
"quiet footsteps",
"golden rays",
"soft whispers",
"heavy clouds",
"broken dreams",
"silent streets",
"quick glance",
"warm hands",
"cold nights",
"dark shadows",
"faint smiles",
"deep waters",
"tall trees",
"hidden secrets",
"empty rooms",
"lonely hearts",
"twinkling stars",
"falling leaves",
"flickering lights",
"distant echoes",
"restless nights",
"forgotten memories",
"brave souls",
"shattered glass",
"dancing flames",
"gentle touch",
"whispered promises",
"velvet curtains",
"ancient ruins",
"mystic forest",
"burning desire",
"chilly winds",
"rolling hills",
"velvet skies",
"snowy peaks",
"rising sun",
"frozen lakes",
"lost time",
"hidden paths",
"distant voices",
"mysterious figures",
"barren lands",
"bright moon",
"old books",
"quiet conversations",
"tangled thoughts",
"silent tears",
"gentle waves",
"midnight dreams",
"rustling leaves",
"fading light",
"crashing waves",
"twisted branches",
"forgotten roads",
"lonely figure",
"raging storm",
"golden sunset",
"cool breeze",
"midnight blue",
"fiery eyes",
"painted skies",
"murmuring streams",
"winding road",
"broken clock",
"silver lining",
"dancing shadows",
"shining stars",
"warm glow",
"rising smoke",
"flickering flame",
"trembling hands",
"echoing footsteps",
"twinkling lights",
"cold dawn",
"hazy memories",
"blinding light",
"silent night",
"bright colors",
"stormy weather",
"gentle breeze",
"worn pages",
"spinning top",
"fragrant flowers",
"creeping fog",
"whispered words",
"soft raindrops",
"endless horizon",
"velvet touch",
"lonely road",
"crimson sky",
"dark night",
"rustling pages",
"soothing music",
"fading echo",
"golden fields",
"midnight hour"
]
words_3_upper = [
"The sky was",
"A gentle breeze",
"She suddenly realized",
"The old man",
"In the distance",
"A loud noise",
"Time passed quickly",
"The sun rose",
"Birds were chirping",
"The rain started",
"They quietly approached",
"He couldn't believe",
"The door opened",
"A strange sound",
"The night was",
"She smiled softly",
"The book fell",
"A shadow moved",
"The wind howled",
"The clock ticked",
"The car sped",
"He slowly turned",
"The lights flickered",
"A sudden gust",
"The phone rang",
"The cat jumped",
"She looked up",
"The water splashed",
"He felt dizzy",
"The train arrived",
"A cold chill",
"The stars twinkled",
"The waves crashed",
"A door creaked",
"The fire crackled",
"He walked away",
"The children laughed",
"The moon shone",
"She whispered softly",
"The road stretched",
"The crowd cheered",
"The sun set",
"A dog barked",
"He stood still",
"The leaves rustled",
"The bell rang",
"She took a",
"He ran fast",
"A soft whisper",
"The wind blew",
"She opened her",
"The tree swayed",
"He reached out",
"The clock struck",
"A sudden noise",
"The floor creaked",
"He sat down",
"The sky darkened",
"A light flickered",
"The crowd gathered",
"The car honked",
"She walked away",
"He picked up",
"A loud bang",
"The clouds parted",
"The rain stopped",
"She felt cold",
"The boat rocked",
"The music played",
"The sun shone",
"A bird flew",
"The door slammed",
"The wind picked",
"He looked around",
"The cat purred",
"The ground shook",
"A light breeze",
"The plane landed",
"He felt tired",
"She stepped forward",
"The lights dimmed",
"The room was",
"A child cried",
"The doorbell rang",
"The fog lifted",
"He heard footsteps",
"The night fell",
"The air was",
"A tear fell",
"The snow fell",
"The water was",
"The car door",
"A soft thud",
"The baby smiled",
"The house was",
"A loud thud",
"The bus stopped",
"The wind died",
"The night sky",
"He leaned forward"
]
words_4_upper=[
"The quick brown fox",
"When the sun sets",
"In the middle of",
"As the day ended",
"Before the meeting started",
"During the summer months",
"After the long journey",
"Across the open field",
"Under the bright sky",
"Over the rolling hills",
"With a gentle touch",
"Beneath the starlit sky",
"Throughout the whole process",
"At the crack of dawn",
"Beyond the city limits",
"Inside the old house",
"Without a second thought",
"By the side of",
"Near the edge of",
"Among the tall trees",
"Within the cozy room",
"Through the dark forest",
"Along the sandy beach",
"Underneath the heavy blanket",
"Inside the busy office",
"Before the final decision",
"Without any hesitation",
"At the top of",
"By the end of",
"Across the narrow bridge",
"Over the crowded street",
"During the late hours",
"With a sudden burst",
"Under the ancient oak",
"Along the winding path",
"Inside the crowded room",
"Before the rain began",
"After the big event",
"At the very start",
"By the old fountain",
"Through the thick fog",
"Under the cool shade",
"Across the quiet lake",
"With a loud cheer",
"Inside the small cabin",
"During the festive season",
"Beyond the distant mountains",
"Without any warning",
"At the base of",
"Near the hidden cave",
"By the warm fire",
"Through the narrow alley",
"Over the frozen lake",
"Along the forest trail",
"Before the new dawn",
"Under the clear water",
"With a gentle smile",
"Inside the ancient ruins",
"Among the scattered leaves",
"Across the vast desert",
"By the quiet river",
"At the edge of",
"During the cold night",
"Without a clear plan",
"Beneath the bright lights",
"After the big storm",
"Along the dusty road",
"Through the old gate",
"With the setting sun",
"Under the thick canopy",
"Inside the grand hall",
"Before the storm hit",
"Among the blooming flowers",
"Near the bustling market",
"By the old bridge",
"During the morning rush",
"Beyond the rolling waves",
"Through the ancient forest",
"With a soft whisper",
"Over the small hill",
"Under the old tree",
"Across the frozen field",
"Inside the wooden cabin",
"Before the start of",
"By the shining moon",
"During the hectic day",
"Without a clear direction",
"At the center of",
"Beneath the cloudy sky",
"With the morning breeze",
"After the brief pause",
"Through the golden fields",
"Over the distant horizon",
"Along the rocky shore",
"Before the sun rises",
"Near the ancient ruins",
"During the festive parade",
"Beneath the moonlit sky",
"Through the open window",
"With the first light"
]
words_4_upper_business=[
"The quarterly report highlights",
"Our team is currently",
"We are pleased to",
"The new marketing strategy",
"Customer satisfaction remains a",
"Our sales figures have",
"This initiative aims to",
"The financial projections indicate",
"Management is focused on",
"Effective communication is essential",
"The company will introduce",
"Recent developments in technology",
"Employee training programs are",
"A comprehensive review of",
"Our client base continues",
"The budget allocation for",
"Strategic partnerships are critical",
"We are expanding our",
"Innovative solutions drive success",
"Market research indicates that",
"The project timeline is",
"This policy will affect",
"Increased investment in infrastructure",
"The new product launch",
"Financial stability is crucial",
"Key performance indicators include",
"Our revenue growth strategy",
"The upcoming conference will",
"Effective leadership is necessary",
"The company’s growth trajectory",
"Operational efficiency must be",
"Our long-term goals focus",
"Recent trends show a",
"Customer feedback is vital",
"The annual general meeting",
"Our competitive advantage lies",
"The team is working",
"Sales targets have been",
"The financial outlook for",
"New regulations impact the",
"Strategic objectives need alignment",
"The performance review process",
"Business development initiatives are",
"Profit margins are expected",
"The market landscape is",
"We are exploring opportunities",
"The proposal outlines several",
"Upcoming projects require significant",
"The company’s mission statement",
"Innovation drives our business",
"The operational challenges faced",
"Project deliverables must meet",
"Current economic conditions affect",
"The team has implemented",
"Our customer retention strategies",
"Financial resources are allocated",
"The annual budget review",
"New technology enhances productivity",
"Performance metrics are evaluated",
"The merger with another",
"We are launching a",
"Key stakeholders must be",
"The revenue model focuses",
"Market penetration strategies include",
"Company policies need updating",
"Our growth strategy emphasizes",
"The risk management framework",
"The team is developing",
"Recent acquisitions expand our",
"Financial reporting requirements include",
"Our competitive positioning strategy",
"New initiatives are designed",
"Business operations are streamlined",
"The customer acquisition process",
"The sales forecast suggests",
"Our investment portfolio consists",
"The strategic plan outlines",
"Financial performance impacts stock",
"The marketing campaign targets",
"Expansion into new markets",
"Key trends affect our",
"The annual report highlights",
"Our service offering includes",
"New software improves efficiency",
"The risk assessment process",
"The operational budget includes",
"Employee engagement drives productivity",
"The merger is expected",
"Strategic investments are planned",
"Financial results will be",
"The cost-benefit analysis reveals",
"Our supply chain management",
"The project scope includes",
"Performance reviews are conducted",
"The sales strategy focuses",
"Key deliverables are outlined",
"The fiscal year ends",
"New policies impact operations",
"Customer acquisition costs are",
"Our business model adapts"
]