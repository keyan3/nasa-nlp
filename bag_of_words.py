import numpy as np
import string, os, time, IPython
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import save_npz
from tfidf import concat_df, check_contiguous

def save_npzs(df, X, string, fns):
    """
    This function takes in a string of column name, and a list of file
    names, retrieve tfidf vector and save a npz file for each fns.
    """
    print('num_files', len(fns))
    for csv_fn in fns:
        fn = string + 'bow_' + csv_fn[-5:-4] + '.npz'
        print('start ' + fn)
        start = time.time()
        df_npy = df[df['filename_clean'] == csv_fn]
        # calculate average cosine similarity for each row
        ##### take off np.mean() for a vector representation for all
        # out = np.array([np.mean(cosine_similarity(X[i], X).flatten()) for i in df_npy.index]) 
        idx = list(df_npy.index)
        assert(check_contiguous(idx))
        out = X[min(idx): min(idx) + len(idx)]
        save_npz(fn, out)
        print('saved ' + fn)
        print("Wall time: {} seconds".format(time.time() - start))

        
def main():     
    in_path = 'datasets/all/' ##### your path to cleaned files
    # Concatenate all cleaned files together
    csv_fns = list(os.listdir(in_path))
    csv_fns = [fn for fn in csv_fns if '.csv' in fn and 'cleaned' in fn]
    df_all = concat_df(csv_fns, in_path)
    
    # Bag of Words on synopsis
    vectorizer_prod = CountVectorizer()
    # Learn vocabulary and idf, return term-document matrix
    X_prod = vectorizer_prod.fit_transform(df_all['synopsis'])
    save_npzs(df_all, X_prod, 'results/synopsis/', csv_fns)
    
    # Bag of Words on narrative
    vectorizer_desc = CountVectorizer()
    # Learn vocabulary and idf, return term-document matrix
    X_desc = vectorizer_desc.fit_transform(df_all['narrative'])
    save_npzs(df_all, X_desc, 'results/narrative/', csv_fns)
    
if __name__ == "__main__":
    main()
