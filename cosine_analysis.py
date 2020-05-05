from numpy.linalg import norm
import numpy as np
import pandas as pd, IPython
from tqdm import tqdm
from scipy.sparse import load_npz
from scipy.sparse.linalg import norm as sparse_norm
import sklearn.metrics
from datetime import datetime, timedelta

# sliding window for cosine similarity calculation. Must be an even number
sliding_window = 2

# helps us print our progress while script runs
column = 0

""" Takes two matrices as an input. In each matrix, each row is a datapoint, and the columns
    represent the vector representation. The matrix a represents a unique entry and therefore
    has only one row. The function computes the average cosine similarities of the datapoint in a 
    to each datapoint in matrix b. For instance, if a has shape (1, 100), there is 1 data point 
    with dimension 100. Consider b with shape (100, 100). This function first calculates 100 individual 
    cosine similarities, one for each row of b, then outputs their average.
    Parameters:
        a: np.ndarray, 1 row
        b: np.ndarray
    Returns:
        float
"""
def average_cosdist(a, b):
    if b.shape[0] == 0:
        return None
    else:
        return np.average(sklearn.metrics.pairwise.cosine_similarity(a, b))
    
def run_compare(pd_text, d2v_data, colname, split = True, notd2v = False):
    assert (pd_text.shape[0] == d2v_data.shape[0])

    """ Returns an ndarray of unique tracon codes in a given dataframe
        Parameters:
            pd_grp_period (pd.DataFrame): any dataframe
        Returns:
            ndarray of unique tracon codes
    """
    def get_unique_tracons(df):
        tracons = df['airporttraconcode1'].unique()
        return tracons

    """ Converts the results from analyze_grp_period and adds a new column in orig
        dataframe (which represents the average cosine similarity of narrative/syn)
        Parameters:
            results: list of ndarrays. Each ndarray represents an entry and has 
                shape (1, 2). The first column includes the average cosine
                similarity whereas the second column indicates the index in the original
                dataframe.
        Returns:
            pd.DataFrame: with added colname representing the average cosine similarity
    """
    def post_process_results(results):
        global column
        results = np.vstack(results)
        sorted_one = np.sort(results[:, 1]) # sort based off of index
        cosdist = results[np.argsort(results[:, 1].flatten()), 0]
        # note that you have to flatten results (otherwise, it argsorts each row)
        if notd2v:
            cosdist = np.asarray(cosdist).flatten()
        assert(pd_text.shape[0] == cosdist.shape[0])
        pd_text[colname] = cosdist
        # print progress %
        column += 1
        print("{:.1f}".format((column / 12) * 100) + "%" + " completed")
        return pd_text

    """ Given a pd.DataFrame which contains all entries for a single tracon, mutates 
        'results' to include ndarrays, each representing a cosine similarity calculation 
        within the tracon (this is post-processed to be added in the original DataFrame via 
        post_process_results:51, see for more details). 

        'Split' arg encodes whether calculations should be done within a sliding window 
        (3mo/6mo/12mo/18mo) s.t. entries are compared to temporally adjacent entries rather 
        than all entries within the tracon.
        Parameters:
            pd_tracon (pd.DataFrame)
            split (boolean)
    """
    def analyze_tracon(pd_tracon, split):

        """ Given a date in the form of pd.Timestamp, calculates the earliest and latest 
            dates that lie within the user-specified sliding window of the input date.
            Parameters:
                curr_date: pd.Timestamp
            Returns:
                (min_date, max_date): Tuple of pd.Timestamp
        """
        def get_min_max_date(curr_date):
            global sliding_window
            window = int(sliding_window / 2)

            # Calculate how many years backwards and forwards we need to look
            neg_year_delta, pos_year_delta = ((curr_date.month - 1) - window) // 12, ((curr_date.month - 1) + window) // 12

            # Find the months of minimum and maximum dates
            min_month, max_month = ((curr_date.month - 1) - window) % 12, ((curr_date.month - 1) + window) % 12

            # Generate the min and max dates 
            min_date = pd.Timestamp(year=curr_date.year + neg_year_delta, month = min_month + 1, day=1)
            max_date = pd.Timestamp(year=curr_date.year + pos_year_delta, month = max_month + 1, day=1)
            return min_date, max_date

        """ Creates an ndarray that includes the average cosine similarity in first column, and
            the index of the entry in the second column (see post_process_results)
            Parameters:
                grouping period (str): time period
            Returns:
                np.ndarray with shape (1, 2)
        """
        def analyze_entry(entry_idx):
            if split:
                curr_date = pd_tracon.loc[entry_idx, 'date']
                min_date, max_date = get_min_max_date(curr_date)
                other_entries_idx = pd_tracon[(pd_tracon['date'] <= max_date) & (pd_tracon['date'] >= min_date)].index
                other_entries_idx = list(other_entries_idx.drop(entry_idx))
            else:
                other_entries_idx = list(pd_tracon.index.drop(entry_idx))

            this, other = d2v_data[entry_idx], d2v_data[other_entries_idx]
            if notd2v:
                cosdist = average_cosdist(this, other)
            else:
                this = np.array([this])
                cosdist = average_cosdist(this, other)
                if cosdist is not None:
                    cosdist = (cosdist + 1) / 2
            return np.array([[cosdist, entry_idx]])

        # adjust results to include all the outputs from analyze_grouping period 
        for entry_idx in pd_tracon.index:
            results.append(analyze_entry(entry_idx))
    
    results = []
    for tracon in get_unique_tracons(pd_text):
        pd_tracon = pd_text[pd_text['airporttraconcode1'] == tracon]
        analyze_tracon(pd_tracon, split)
    return post_process_results(results)

def create_metadata_obj(str_rep):
    str_reps = ['d2v', 'tfidf', 'bow']
    colnames = ['{}_cossim_nar', '{}_cossim_syn']
    colnames = colnames + [elem + "_all" for elem in colnames]
    folders = ['narrative', 'synopsis']
    assert (str_rep in str_reps)
    for colname_ind in range(len(colnames)):
        if str_rep == "d2v":
            yield ('results/{}/{}_{}.{}'.format(folders[colname_ind % 2], str_rep, '{}', 'npy'),
                    colnames[colname_ind].format(str_rep))
        else:
            yield ('results/{}/{}_{}.{}'.format(folders[colname_ind % 2], str_rep, '{}', 'npz'),
                    colnames[colname_ind].format(str_rep))

""" This analyzes all the dataframes/doc2vec representations saved via
    generate_doc2vec:clean_create_doc2vec.
    Parameters:
        save_dfs (bool): whether or not to save the final_cosine_df
            for each particular file. The whole dataframe (concatenated) is saved regardless
"""

def analyze_cosine(save_df = False):
    import os
    pd_files = 'datasets/all/cleaned_df_{}.csv'

    dataset_fn = 'datasets/asrs/'
    csv_fns = list(os.listdir(dataset_fn))
    csv_fns = [fn for fn in csv_fns if '.csv' in fn]
    num_files = len(csv_fns)

    # read cleaned df and create datetime column for sliding window analysis
    df = pd.read_csv(pd_files.format(0), index_col = 0)
    df.reset_index(inplace = True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df = df.rename(columns = {'index':'orig_index'})

    for str_rep in ['d2v', 'tfidf', 'bow']:
        for npy_fn, colname in create_metadata_obj(str_rep):
            if str_rep == "d2v":
                curr_npy = np.load(npy_fn.format(0))
            else:
                curr_npy = load_npz(npy_fn.format(0))
            assert(df.shape[0] == curr_npy.shape[0])
            df = run_compare(df, curr_npy, colname, split = "_all" not in colname,
                    notd2v = str_rep != "d2v")

    os.makedirs(os.path.dirname('results/all/final_cosine_df.csv'), exist_ok=True)
    df.to_csv('results/all/final_cosine_df.csv')

if __name__ == "__main__":
    analyze_cosine(save_df = False)
