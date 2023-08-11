import streamlit as st
import pandas as pd
import numpy as np

# Create a title and sub-title
st.header("LatinCy lg similarity search")

LENGTH = 25

#Load and cache data with streamlit decorator
@st.cache_resource
def load_data():
    df = pd.read_csv('models/vectors-50000.zip', compression='zip', sep='\t', index_col=0, header=None)
    df.index.name = 'word'
    return df

# Create pull down menu from df index
df_vectors = load_data()
query = st.selectbox("Term", sorted(df_vectors.index.tolist()), index=0)

if query:
    st.write(f"Previewing the top {LENGTH} most similar terms to '{query}'. Click button to download full list")
    df_query = df_vectors.loc[query]
    df_diff = df_vectors - df_query.values
    df_norm = df_diff.apply(np.linalg.norm, axis=1)
    idx = np.argsort(df_norm, axis=1)
    results = df_norm.iloc[idx]
    df_results = pd.DataFrame(results, columns=['l1_distance']).reset_index()
    df_results.index.name = 'rank'
    df_display = df_results[:LENGTH]
    st.dataframe(df_display, width=300)

    # Download button
    output = df_results.to_csv(sep='\t')
    # output.index.name='rank'
    st.download_button("Download", data=output, file_name=f"lg_sims-{query}.tsv", mime='text/tab-separated-values')
