import numpy as np
import string, os, time
import pandas as pd
import IPython
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import save_npz


def concat_df(fns, in_path):
    """
    This function takes in a list of file names in .csv and a path to 
    files. It reads the files and then concat all files into one single 
    dataframe.
    
    Input:
        fns -> list of str OR str
        in_path -> str
    Output:
        df -> pd.DataFrame
    """
    df = pd.DataFrame()
    # check if input fns is a single file or a list of filenames
    if type(fns) != list:
        total_fn = in_path + fns
        df = pd.read_csv(total_fn, index_col=0)
        df['filename_clean'] = fns
        return df
    for csv_fn in fns:
        total_fn = in_path + csv_fn
        df_clean = pd.read_csv(total_fn, index_col=0)
        # when read each csv, create a column to save current csv name
        df_clean['filename_clean'] = csv_fn
        df = pd.concat([df, df_clean]) 
    df = df.reset_index(drop=True)
    return df

def check_contiguous(lst):
    """
    This function check if the list is contiguous or not.
    
    Input:
        lst -> list of numbers
    output: 
        bool
    """
    for i in range(1, len(lst)):
        if (lst[i] != lst[i - 1] + 1):
            return False
    return True

def save_npzs(df, X, path, fns):
    """
    This function takes in a string of column name, and a list of file
    names, retrieve tfidf vector and save a npz file for each fns.
    
    Input:
        df -> pd.DataFrame
        X -> scipy.csr.csr_matrix
        path -> str
        fns -> list of str OR str
    Output:
        None
    """
    def save_helper(fns):
        """
        This helper function takes in a single filename and 
        save a npz file.
        
        Input:
            fns -> list of str OR str
        Output:
            None
        """
        fn = path + 'tfidf_' + fns[-5:-4] + '.npz'
        print('start ' + fn)
        start = time.time()
        df_npy = df[df['filename_clean'] == fns]
        # calculate average cosine similarity for each row
        idx = list(df_npy.index)
        assert(check_contiguous(idx))
        out = X[min(idx): min(idx) + len(idx)]
        save_npz(fn, out)
        print('saved ' + fn)
        print("Time: {} seconds".format(time.time() - start) + '\n')
    # check if input fns is a single file or a list of filenames
    if type(fns) != list:
        print('num_files: 1')
        print("======================================================")
        save_helper(fns)
    else:
        print('num_files:', len(fns))
        print("======================================================")
        for csv_fn in fns:
            save_helper(csv_fn)

def save_tfidf(df, col_name, save_path, fns):
    """
    This function takes in a dataframe, and fit a tfidf vectorizer on 
    the column name specified, then save to save_path.
    
    Input:
        df -> pd.DataFrame
        col_name -> str
        save_path -> str
        fns -> list of str OR str
    Output:
        None
    """
    print("start " + col_name + '...')
    # Initialize vectorizer
    vectorizer = TfidfVectorizer()
    # Learn vocabulary and idf, return term-document matrix for col_name
    X = vectorizer.fit_transform(df[col_name])
    # Save the model as npz file
    save_npzs(df, X, save_path, fns)
    print("======================================================")
    print('finished and saved ' + col_name + '\n')

        
def main():     
    in_path = 'datasets/all/' ##### your path to cleaned files
    # Concatenate all cleaned files together
    csv_fns = list(os.listdir(in_path))
    csv_fns = [fn for fn in csv_fns if '.csv' in fn and 'cleaned' in fn]
    df = concat_df(csv_fns, in_path)
    
    # Tf-idf on synopsis
    save_tfidf(df, 'synopsis', 'results/synopsis/', csv_fns)
    # Tf-idf on narrative
    save_tfidf(df, 'narrative', 'results/narrative/', csv_fns)

    
if __name__ == "__main__":
    main()
