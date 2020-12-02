import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import plotly as py
import plotly.graph_objs as go

st.title('Country Clusters')

country = pd.read_csv(r'C:\Users\DELL PC\PythonFiles\Machine Learning\Clustering\Assignment\Country-data.csv')

def df_attr(df):
    count = df.count()
    null = df.isnull().sum()
    null_perc = round(df.isnull().sum()/len(df.index)*100,4)
    unique = df.nunique()
    types = df.dtypes
    desc = df.describe().T
    return pd.concat([count, null, null_perc, unique, types, desc], axis = 1,
                     keys=['COUNT','NULL','PERCENT','NUM_UNIQUE','DATATYPE'])

d = {
    'child_mort':'Child Mortality',
    'exports':'Exports',
    'imports':'Imports',
    'health':'Health expenses',
    'income':'Net income per capita',
    'inflation':'Inflation',
    'life_expec':'Life expectancy',
    'total_fer':'Fertility rate',
    'gdpp':'GDP per capita'
    }

# Changing the relative values to absolute values
country['exports'] = country['exports']*country['gdpp']/100
country['imports'] = country['imports']*country['gdpp']/100
country['health'] = country['health']*country['gdpp']/100

if st.checkbox('Show data'):
    st.markdown('#### Data')
    st.write(country)

    shape = country.shape
    st.write(f'The data has {shape[0]} rows and {shape[1]} columns.')

    country_attr = df_attr(country)
    st.markdown('#### Data Info')
    st.write(country_attr)

    st.markdown('#### Data Summary')
    st.write(country.describe().T)

# Plotly choropleth
st.markdown('#### Data Visualisation')
a = st.selectbox('Select variable', country.columns[1:])

if a in ['child_mort', 'total_fer', 'inflation']:
    cmap = 'Reds'
elif a in ['income', 'health', 'life_expec']:
    cmap = 'Greens'
else:
    cmap = 'Blues'

fig = go.Figure(data=go.Choropleth(locations=country['country'], locationmode='country names',
                                   z=country[a], colorscale=cmap,
                                   marker_line_color='black', marker_line_width=0.5))
fig.update_layout(title_text=f'{d[a]} of Countries', title_x=0.5,
                  geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'))
st.plotly_chart(fig)

# Dropping 'country' before analysis
df = country.drop('country', axis=1).copy()

# Capping only the upper outliers at 95th percentile for all the variables
for col in df.columns:
    Q = df[col].quantile(0.95)
    df[col] = np.where(df[col] > Q, Q, df[col])

# Scaling the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Clustering selection
st.markdown('### Select clustering method')
clus = st.radio('Select clustering method', ['K-Means', 'Hierarchical'])

if clus == 'K-Means':
    st.markdown('## **K-Means Clustering**')
    st.markdown('⬅️ Controls are on the sidebar')
    st.markdown('')

    # k selection
    st.sidebar.title('K-Means Clustering Controls')
    st.sidebar.markdown('#### Select the number of clusters')
    k = st.sidebar.number_input('Select n clusters: 2 to 10', 2, 10, value=3)
    st.write(f'Number of clusters selected: **{k}**')

    # K-means
    kmeans = KMeans(n_clusters=k, max_iter=50, random_state=100)
    kmeans.fit(df_scaled)

    ssd = round(kmeans.inertia_,4)
    cluster_labels = kmeans.labels_
    silhouette_avg = round(silhouette_score(df_scaled, cluster_labels),4)
    st.write(f'Sum of squared distances: **{ssd}**')
    st.write(f'Silhouette Score: **{silhouette_avg}**')

    # Adding cluster_id to original df
    df_clusters = country.copy()
    df_clusters['cluster_id'] = kmeans.labels_

elif clus == 'Hierarchical':
    st.markdown('## **Hierarchial Clustering**')
    st.markdown('⬅️ Controls are on the sidebar')

    st.markdown('')

    st.sidebar.title('Hierarchial Clustering Controls')
    st.sidebar.markdown('#### Select linkage')
    link = st.sidebar.radio('Select linkage', ['single', 'complete'], 1)

    st.markdown(f'### **{link.title()} Linkage**')

    @st.cache
    def hier(df_scaled, link):
        merg = linkage(df_scaled, method=link, metric='euclidean')
        return merg

    merg = hier(df_scaled, link)
    dendrogram(merg)
    st.pyplot(use_container_width=True)

    st.sidebar.markdown('#### Select the number of clusters')
    n = st.sidebar.number_input('Select n clusters', 2, 10, value=2)
    st.write(f'Number of clusters selected: **{n}**')

    # Cutting the complete_mergings into 3 clusters
    cluster_labels = cut_tree(merg, n_clusters=n).reshape(-1, )

    df_clusters = country.copy()
    df_clusters['cluster_id'] = cluster_labels

# Selecting variables to be plotted
cols = df.columns.to_list()

st.sidebar.markdown('#### Select variable 1')
var_x = st.sidebar.selectbox('Select x for the plot', cols, 8)
st.sidebar.markdown('#### Select variable 2')
var_y = st.sidebar.selectbox('Select y for the plot', cols, 4)

st.markdown('### Clusters of countries')
if var_x != var_y:
    st.markdown(f'#### {d[var_x]} vs. {d[var_y]}')
    st.markdown('')
    chart = alt.Chart(df_clusters).mark_circle(size=75).encode(
        x=var_x,
        y=var_y,
        color='cluster_id:N',
        tooltip=['country'],).\
        properties(height=400).\
        configure_axis(grid=False).\
        interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning('Please select different variables.')
