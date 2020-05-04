import pandas as pd, numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import gensim, os, IPython, pickle, time
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
english_stopwords = set(stopwords.words('english'))
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import uuid
from collections import defaultdict

""" This is Doc2Vec Specific. This creates an iterator using files, saved to disk
    of all the sentences needed to train Doc2Vec Model. Sentences are lists of words
    ['this', 'is', 'an' 'example']. This assumes that there are pickle files saved in
    a particular format (e.g.: 'datasets/file_{}.pckl' where {} is an index from 0 to
    the number of files). Created to save RAM and not destroy your computer. """
class DocSentences:
    """ Parameters:
            in_format: a format string, where the input is a num (e.g.: 'datasets/sent/{}.pckl')
            num_fns: number of files in that particular format
    """
    def __init__(self, in_format, num_fns = 23):
        self.in_format = in_format
        self.num_fns = num_fns
        self.len = None
        self._reset()

    """ Resets variables that keep track of which file you're currently at
    """
    def _reset(self):
        self.file_idx = 0 # which file you're on
        self.total_idx = 0 # total index
        self.infile_idx = 0 # index of the file you're on
        self.infile_len = None # length of file you're on
        self.tmp_sentences = None # list of current senteces
        self.sentences_dict = {} # dictionary from sentence to index

    """ Checks if the current iterator is at the end of a particular pickle file
    """
    def _at_end_of_file(self):
        assert(self.infile_len is not None)
        assert(len(self.tmp_sentences) == self.infile_len)
        return self.infile_idx == self.infile_len

    """ Loads the current file """
    def _load_curr_file(self):
        self.tmp_sentences = pickle.load(open(self.in_format.format(str(self.file_idx)), "rb"))
        self.infile_len = len(self.tmp_sentences)
        self.infile_idx = 0

    """ Loads next file"""
    def _load_next_file(self):
        self.file_idx += 1
        self._load_curr_file()

    """ Given a sentence, give the corresponding index. Doc2vecs saves all the vectors,
        by indexing the all the sentences. This provides us an easy way to get the idx from
       a sentence """
    def get_sentence_idx(self, sentence):
        if self.sentences_dict and sentence in self.sentences_dict:
            return self.sentences_dict[sentence]
        else:
            pass

    """ Check if you are at the end of a file, and iterate to next if you are """
    def _check_and_continue(self):
        if self._at_end_of_file():
            if self.file_idx != self.num_fns - 1:
                self._load_next_file()
            else:
                raise StopIteration()
    def __iter__(self):
        return self

    def __next__(self):
        # start up code for tmp sentences
        if not self.tmp_sentences:
            self._load_curr_file()
        self._check_and_continue()

        # create the string from a list of words and continue until unique
        doc = ' '.join(self.tmp_sentences[self.infile_idx])
        self.infile_idx += 1
        while (doc in self.sentences_dict): # this skips duplicates
            self._check_and_continue()
            doc = ' '.join(self.tmp_sentences[self.infile_idx])
            self.infile_idx += 1

        self.sentences_dict[doc] = self.total_idx
        return_val = TaggedDocument(doc, [self.total_idx]) 
        self.total_idx += 1
        return return_val # what the Doc2vec model is expecting

""" Lowers the string for each given column
    Parameters:
        av_text (pd.DataFrame): raw dataframe
        cols (list of str): list of column names to be lowerd
    Returns:
        av_text (pd.DataFrame): adjusted dataframe
"""
def lower_cols(av_text, cols = ['narrative', 'synopsis', 'airporttraconcode1']):
    for col in cols:
        av_text[col] = av_text[col].str.lower()
    return av_text

""" Given a particular string, remove stopwords, and stem/lemmatize.
    Parameters:
        sent (str): sentence to be processed
    Returns:
        sent (str): processed string
"""
def pre_process_str(sent):
    fin_lst = []
    for word in sent.split():
        if word not in english_stopwords:
            fin_lst.append(stemmer.stem(wnl.lemmatize(word.lower())))
    return ' '.join(fin_lst)

""" Cleans a pd.Series (str) by removing extraneous characters and preprocessing
    Parameters:
        str_series (pd.Series): the panda series you would like to clean
    Returns:
        str_series (pd.Series): cleaned series
"""
def create_clean_str_series(str_series):
    for char in '[!"#$&\'()*+,:;<=>?@[\\]^_`{|}~/-]\.%':
        str_series = str_series.str.replace(char, '')
    str_series = str_series.str.strip()
    str_series = str_series.apply(pre_process_str)
    return str_series

""" Convert pd.Series (str) into list of lists of str. Each list represents a sentence (narrative/synopsis)
    and each str represents a word (within a narrative/synopsis)
    Parameters:
        str_data (pd.Series): the panda series containing 
    """
def generate_sentences(str_data):
    sentences = []
    for elem in str_data:
        if type(elem) is float: #TODO, everything should be string already, fix earlier
            elem = str(elem)
        sentences.append(elem.split(" "))
    return sentences

def clean_and_create_sentences(dataset_fn, csv_fns):
    for idx, csv_fn in tqdm(enumerate(csv_fns)):
        total_fn = dataset_fn + csv_fn
        av_text = pd.read_csv(total_fn)

        print("\nBeginning cleaning")

        # drop na values   
        av_text = av_text.loc[av_text['narrative'].dropna().index]
        av_text = av_text.loc[av_text['synopsis'].dropna().index]
        av_text = av_text.loc[av_text['airporttraconcode1'].dropna().index]
        av_text = av_text.loc[av_text['date'].dropna().index]

        # lower all string columns
        av_text = lower_cols(av_text) 
        av_text['narrative'] = av_text['narrative'].astype(str)
        av_text['synopsis'] = av_text['synopsis'].astype(str)
        av_text['date'] = av_text['date'].apply(int).astype(str)

        # clean date
        av_text = av_text.loc[av_text['date'].str.len() == 6]

        # clean narrative
        clean_narrative = create_clean_str_series(av_text['narrative'])
        clean_narrative.dropna(inplace = True)
        av_text['clean_narrative'] = clean_narrative
        av_text = av_text[av_text['clean_narrative'] != '']

        # clean synopsis
        clean_synopsis = create_clean_str_series(av_text['synopsis'])
        clean_synopsis.dropna(inplace = True)
        av_text['clean_synopsis'] = clean_synopsis
        av_text = av_text[av_text['clean_synopsis'] != '']

        # handle pandas bug with nan string literal (as opposed to NaN type)
        av_text = av_text[av_text['airporttraconcode1'] != 'nan']

        # flag dups
        synopsis_dict = defaultdict(lambda:0)
        for i in av_text.index:
            synopsis_dict[(av_text.loc[i, 'narrative'], av_text.loc[i, 'airporttraconcode1'])] += 1
        is_dup = []
        for i in av_text.index:
            if synopsis_dict[(av_text.loc[i, 'narrative'], av_text.loc[i, 'airporttraconcode1'])] > 1:
                is_dup.append(1)
            else:
                is_dup.append(0)
        av_text['is_dup'] = is_dup

        print("Beginning sentence generation")

        # generate sentences
        narratives = generate_sentences(clean_narrative.values)
        synopses = generate_sentences(clean_synopsis.values)

        # save sentences to specific folders
        list_data = [narratives, synopses]
        savefn_list = ['datasets/narrative/{}.pckl'.format(str(idx)), 'datasets/synopsis/{}.pckl'.format(str(idx))]
        for list_idx in range(len(list_data)):
            os.makedirs(os.path.dirname(savefn_list[list_idx]), exist_ok=True)
            pickle.dump(list_data[list_idx], open(savefn_list[list_idx], "wb"))

        os.makedirs(os.path.dirname('datasets/all/cleaned_df_{}.csv'.format(str(idx))), exist_ok=True)
        av_text.to_csv('datasets/all/cleaned_df_{}.csv'.format(str(idx))) # this needs to include all rows

def create_doc2vec(num_files, vector_size, min_count, epochs):
    output_type = ['narrative', 'synopsis']
    in_formats = ["datasets/{}/{}.pckl".format(elem, "{}") for elem in output_type]
    d2v_save = ["models/narratives.model", "models/synopses.model"]
    out_formats = ["results/{}/d2v_{}.npy".format(elem, "{}") for elem in output_type]
    colnames = ["clean_narrative", "clean_synopsis"]
    for i in range(0, 2):

        ds = DocSentences(in_formats[i], num_fns = num_files)
        pickle.dump(ds, open("datasets/all/ds_{}.pckl".format(output_type[i]), "wb"))
        d2v_model = Doc2Vec(ds, vector_size = vector_size, min_count = min_count)
        d2v_model.train(ds, total_examples = d2v_model.corpus_count, epochs = epochs)

        # save doc2vec model just in case
        fname = get_tmpfile(os.path.abspath(d2v_save[i]))
        d2v_model.save(fname)

        for file_idx in tqdm(range(num_files)):
            pd_file = pd.read_csv('datasets/all/cleaned_df_{}.csv'.format(str(file_idx)), index_col = 0)
            d2v_np = np.zeros((pd_file.shape[0], vector_size))
            for idx in range(pd_file.shape[0]):
                sentence = pd_file.iloc[idx][colnames[i]]
                if (type(sentence) is float): # TODO: not sure why this is happening
                    sentence = str(sentence)
                sentence_idx = ds.get_sentence_idx(sentence)
                if sentence_idx is not None:
                    d2v_np[idx] = d2v_model.docvecs[sentence_idx]
                else:
                    raise Exception("'{}' not found".format(sentence))
            os.makedirs(os.path.dirname(out_formats[i].format(str(file_idx))), exist_ok=True)
            np.save(out_formats[i].format(str(file_idx)), d2v_np)

def clean_create_doc2vec(vector_size = 100, min_count = 10, epochs = 20):
    # d2v_params: vector_size, min_count, epochs
    # save_d2v_models (T/F):
    dataset_fn = 'datasets/asrs/'
    csv_fns = list(os.listdir(dataset_fn))
    csv_fns = [fn for fn in csv_fns if '.csv' in fn]
    num_files = len(csv_fns)

    clean_and_create_sentences(dataset_fn, csv_fns)
    print("Beginning doc2vec generation")
    create_doc2vec(num_files, vector_size, min_count, epochs)

if __name__ == "__main__":
    clean_create_doc2vec()
