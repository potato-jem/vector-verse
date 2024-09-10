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
nltk.download('stopwords')
# Get the list of stop words
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(string.punctuation))
common_words=pd.read_csv('./dependencies/freq.csv')
common_words=common_words.set_index("lemma")
common_words.index.name='words'
common_words=common_words[~(common_words.index.isin(stop_words) | (common_words["PoS"]=="x"))]
common_words=list(common_words.index)
#%%
def get_example_pairs(input_text,
                      max_total_tokens=15,
                      temperature=1,
                      top_logprobs=3,
                      system_instruction="continue the text",
                      category=""):
    
    input_length=len(tokenizer.encode(input_text))
    responseX = client.chat.completions.create(
        model="gpt-4o-mini"
        ,messages=[
            {"role": "system",
            "content": system_instruction},
            {
            "role": "assistant",
            "content": input_text
            }
            ]
        ,max_tokens=max_total_tokens-input_length
        ,temperature=temperature
        ,logprobs=True
        ,top_logprobs=top_logprobs
    )

    cumulative_message = input_text+" "
    cumulative_token_length = responseX.usage.prompt_tokens-11-len(tokenizer.encode(system_instruction))
    fingerprint=responseX.system_fingerprint
    df=pd.DataFrame()
    for tkn_info in responseX.choices[0].logprobs.content:
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
        dfp["id"] = hash(fingerprint+cumulative_message)
        dfp["fp"] = fingerprint
        dfp["usage"] = responseX.usage.total_tokens
        dfp["category"]=category
        df=pd.concat([df,dfp])
        cumulative_message+=tkn_info.token
        cumulative_token_length+=1
    print(responseX.usage.total_tokens)    
    df=df.reset_index(drop=True)
    return(df)
#%%
sentence_starters=[]
#sentence_starters=common_words[1:20]
#df=pd.concat([get_example_pairs(x,temperature=1) for x in sentence_starters])
df2u=pd.concat([get_example_pairs(x,temperature=1,category="2upper") for x in words_2_upper])
df2l=pd.concat([get_example_pairs(x,temperature=1,category="2lower") for x in words_2_lower])
df3u=pd.concat([get_example_pairs(x,temperature=1,category="3upper") for x in words_3_upper])
df4u=pd.concat([get_example_pairs(x,temperature=1,category="4upper") for x in words_4_upper])
df4u_business=pd.concat([get_example_pairs(x,temperature=1,category="4upper_business") for x in words_4_upper_business])
df=pd.concat([df2u,df2l,df3u,df4u,df4u_business])
df2u.to_csv("outputs/run1.csv",index=False)
#%%
df["classification"]=""
df["token1_stripped"]=df["token1"].str.strip().str.lower()
df["token2_stripped"]=df["token2"].str.strip().str.lower()
potential_condition=((df["cum_prob2"]>0.8)  
     & (df["prob1"]<0.6)  
     & ~(df["token1_stripped"].isin(stop_words)) 
     & ~(df["token2_stripped"].isin(stop_words))  
     & (df["token1_stripped"].str.len()>=4) 
     & (df["token1_stripped"].str.len()>=4)) 
df.loc[potential_condition,"classification"] = "potential"
df=pd.merge(df, x, how='left', on=["token1_stripped","token2_stripped"],suffixes=["",".new"])
df["classification"]=df["classification.new"].fillna(df["classification"])
df=df.drop("classification.new",axis=1)
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
for row in df[df["classification"]!=""].itertuples():
    #print(row.token1.strip())
    data = {
        'target1':row.token1,
        'target2':row.token2,
        'prob1':row.prob1,
        'prob2':row.prob2,
        'classification': row.classification,
        'category':row.category,
        'answer': row.context,
        'difficulty': 0,
        'stale': False,
        'user_count': 0,
    }
    ref = db.collection(f"targets/TEST/{row.classification}_items")
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