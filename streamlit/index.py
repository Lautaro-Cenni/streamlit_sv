import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
le = LabelEncoder()

df_show =  pd.read_csv('../datasets_streamlit/df_show.csv')
df =  pd.read_csv('../datasets_streamlit/df.csv')
df_test =  pd.read_csv('../datasets_streamlit/df_test.csv')

N = df['customer_le'].unique().shape[0]
M = df['product_le'].unique().shape[0]

#creo una matriz que contiene todas las opiniones de todos los productos por todos los custemers, a partir
#de esto entrenare a mi modelo para que analise similitudes entre los vectores
X = csr_matrix((df["review_score"].values, (df['product_le'].values, df['customer_le'].values)),shape=(M,N))

KNN = NearestNeighbors(n_neighbors=10,algorithm='brute',metric='cosine')
KNN.fit(X)
def find_similar_products(product_id,matrix,k):
    '''
    se le proporciona el id de un producto y te devuelve un listado de productos similares incluyendo el mismo,
    la metrica predeterminada es cosine
    '''
    neighbours_ids= []
    
    product_vector = X[product_id]
    if product_vector.size > 0:
        neighbours = KNN.kneighbors(product_vector, return_distance = True)
        for i in range(1,k+1):
            if neighbours[0].item(i) < 0.35:
                neighbours_ids.append(neighbours[1].item(i))
            
        if len(neighbours_ids) > 0:
            return neighbours_ids
    return None

st.title("Modelo de Machine Learning: Sistema de recomendaciones")
st.markdown('###')
st.dataframe(df_show)
product_label_encoder = st.number_input("ingrese el numero del producto",step=1,format="%i")
if product_label_encoder:
    list_recomended = find_similar_products(product_label_encoder,X,3)
    if type(list_recomended) == list:
        st.write("productos parecidos:")
        st.dataframe(df_test.loc[df_test['product_le'].isin(list_recomended)])
    else:
        st.write("no hay conincidencias")