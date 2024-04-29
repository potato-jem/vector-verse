
#%%
from astropy.time import Time,TimeDelta
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dateutil.parser
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
planets=(
    ('sun',0),
    ('mercury',87.97),
    ('venus',224.7),
    ('earth',365.26),
    ('mars',686.6888),
    ('jupiter',4331.9836),
    ('saturn',10760.5596),
    ('uranus',30685.4926),
    ('neptune',60191.195)
 )
color_map= {
'sun':'yellow',
'mercury':'dimgrey',
'venus':'yellowgreen',
'earth':'blue',
'mars':'red',
'jupiter':'darkred',
'saturn':'goldenrod',
'uranus':'darkslateblue',
'neptune':'darkolivegreen'
}

# %%
locations={}
t = Time("2022-06-19 21:53")
current_earth_coords=get_body_barycentric('earth', t) 
with solar_system_ephemeris.set('builtin'):
    for body,period in planets:
        time_series=[ t+TimeDelta(x*period/200) for x in range(-100,101)]
        p=-100
        for t in time_series:
            coords=get_body_barycentric(body, t) 
            # coords=get_body(body,t)
            # dateutil.parser.parse(t.value).year
            locations[(body,p)]=coords._xyz.value
            p+=1


df=pd.DataFrame(locations,index=['x','y','z']).T
df=df.reset_index().rename(columns={'level_0':'name','level_1':'period'})
df['label_name']=df['name']
df.loc[df['period']!=2022,'label_name']=""
df['size']=0.03
df.loc[df['period']==2022,'size']=1

X = df[['x','y','z']].to_numpy()
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
df[['x_3d','y_3d','z_3d']]=df[['x','y','z']]
df[['x_2d','y_2d']]=X_pca
df['z_2d']=0
df[['x','y','z']]=df[['x_2d','y_2d','z_2d']]
#%%
# fig=px.scatter(df,x='x',y='y',size='size',color='name',text='label_name')
filter=df['period']==0

# fig=px.scatter_3d(df[filter],x='x_2d',y='y_2d',z='z_2d',size='size',color='name',text='label_name')
fig=px.line_3d(df[~filter],x='x',y='y',z='z',color='name',color_discrete_map=color_map)
fig.update_traces(line=dict(width=0.25),hoverinfo='none',hovertemplate=None)
for body,period in planets:
    filter2=(df['period']==0)&(df['name']==body)
    fig.add_trace(go.Scatter3d(x=df[filter2]['x'],y=df[filter2]['y'],z=df[filter2]['z'],mode='markers',name=body,marker={'color':color_map[body]}))

max_axis_list=df[['x','y','z']].max().to_list()
def to_eye_coords(input):
    return dict(zip(['x','y','z'],input/max_axis_list))
zoom=0.03
earth_eye=to_eye_coords(current_earth_coords._xyz.value)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=zoom, y=0, z=3*zoom)
)

fig.update_layout(scene_camera=camera,
    scene=dict(
        xaxis = dict(nticks=4, range=[-max_axis_list[0],max_axis_list[0]],),
                     yaxis = dict(nticks=4, range=[-max_axis_list[1],max_axis_list[1]],),
                     zaxis = dict(nticks=4, range=[-max_axis_list[2],max_axis_list[2]],),),
    width=700,
    margin=dict(r=0, l=0, b=0, t=0))
fig.show()

# %%
