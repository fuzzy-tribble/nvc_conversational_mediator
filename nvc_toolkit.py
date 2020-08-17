import nltk
from nltk.stem import PorterStemmer
import spacy
from pattern.text.en import singularize, lemma, lexeme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import re, time, os, itertools
from typing import Tuple, Any, Optional

nlp = spacy.load("en")
ps = PorterStemmer()

# Feedback categories
fb_cats = ['observations', 'feelings', 'needs', 'needs_guesses', 'requests', 'fofeelings', 'thoughts', 'evaluations', 'absolutes']

def load_dfs(kwp_datafile_relpath:str="data/kwp_datafile.csv") -> pd.DataFrame:
    """Load, clean and verify datafiles
    """
    df = None
    dirname = os.path.dirname(__file__)
    pth = os.path.join(dirname, kwp_datafile_relpath)
    df = pd.read_csv(pth)
    for r in df.index:  # make a lemmas column
        df.loc[r,'kwp_lemma'] = kwp_lemma(df.loc[r,'kwp'])
    df = clean_df(df)
    # TODO (high) - Datachecks - something can't be a feeling and a fofeeling, etc, etc
    return df


def clean_df(df:pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the input data
    lowercase, remove dups
    """
    # lowercase columns and index if str
    if df.columns.dtype == object: df.columns = df.columns.str.lower()
    if df.index.dtype == object: df.index = df.index.str.lower()
    # lowercase rest of the data
    # TODO (optional) - this is a really ugly way of doing this...better way?
    # yes, use applymap(func) for element-wise 
    for ci in range(len(df.columns)):
        if df.iloc[:,ci].dtype == object: # non-numerics are 'objects' in pd.dtypes
            if type(df.iloc[0,ci]) == str:
                df.iloc[:,ci] = df.iloc[:,ci].str.lower()
            elif type(df.iloc[0,ci]) == list:
                # conv lists to strings (pandas works better with strings)
                df.iloc[:,ci] = df.iloc[:,ci].apply(lambda x: ','.join(sorted([str(i).lower() for i in x])))

    # TODO - 'category' column can't be empty

    # replace na with empty (better readability)
    df = df.replace(np.nan, '', regex=True)

    # remove duplicates
    # TODO (crit) - fix me
    # df_clean = df.drop_duplicates()
    return df


def kwp_lemma(kwp:str)->str:
    """Generate lemma version of key word or phrase
    """
    words = kwp.split()
    for i in range(len(words)):
        # Temp fix for upen bug rept: https://github.com/clips/pattern/issues/243
        try: lemma(words[i])
        except: pass
        words[i] = lemma(words[i])
    return ' '.join(words)


def get_raw_feedback(inp:str) -> pd.DataFrame:
    """Get nvc feedback from input sentence
    """
    tool_res = compare_tools(inp)
    tool_res = tool_res.loc[:, tool_res.columns != 'accur_probability']
    compressed_res_list = []
    for row in tool_res.index:
        m = tool_res.loc[row, tool_res.loc[row,:].notnull()].unique()
        compressed_res_list.append(','.join([i for i in m if i]))
    compressed_res = pd.DataFrame(data=[compressed_res_list], columns=fb_cats, dtype=object)
    compressed_res = compressed_res.replace(np.nan, '', regex=True)
    return compressed_res


def compare_tools(inp:str, threshold:float=.5) -> pd.DataFrame:
    """Get nvc feedback from input sentence
    """
    parsed_df = parse_sent(inp)
    tool_results = []
    kwp_df = load_dfs()
    # Run using tool#1 - kwp lookups
    tool_results.append(find_kwp_matches(parsed_df, kwp_df))
    # Run using tool#2 - pos tags
    tool_results.append(find_pos_matches(parsed_df))
    # TODO - add more tools?
    res = pd.concat(tool_results, axis=1, join="outer")
    # Score the tools based on what they found
    # TODO - figure out how to score them
    # TODO - figure out if I want ot mash results into a final series with threshold possibility
    res['accur_probability'] = 0
    res = res.replace(np.nan, '', regex=True)
    # df.applymap(lambda x: ','.join(sorted(x.split(','))) if x!='' else x)
    return res


def parse_sent(inp:str) -> pd.DataFrame:
    """Parse the input sentence
    """
    doc = nlp(inp)
    words = [tok.text for tok in doc]
    # TODO (opt) - allow for custom stemmer or tagging
    data = [[tok.lemma_, kwp_lemma(tok.text), ps.stem(tok.text), tok.pos_, tok.tag_, tok.dep_] for tok in doc]
    return pd.DataFrame(index=words, data=data, columns=['lemma', 'lemma_alt', 'stem', 'pos', 'tag', 'dep'])


def find_kwp_matches(parsed_sent_df:pd.DataFrame, kwp_df:pd.DataFrame) -> pd.DataFrame:
    """Get keywords and phrase matches
    """
    if kwp_df.empty: raise(Exception("kwp_df must not be empty."))
    kwp_matches = pd.DataFrame(index=fb_cats, dtype=object)
    # TODO - populate sent_lemma_list only with non punctuation, non interjections, etc
    sent_lemma_list = parsed_sent_df.loc[:,'lemma'].values
    for i in range(len(sent_lemma_list)):
        for kwp_lemma in kwp_df.kwp_lemma.values:
            kwp_lemma_list = kwp_lemma.split()
            if len(kwp_lemma_list) > 1:  # its a phrase
                if ( kwp_lemma_list[0] == sent_lemma_list[i] and 
                    len(sent_lemma_list[i:]) >= len(kwp_lemma_list) ):
                    # first word matches and 
                    # sent has enough length to make the phrase
                    # print(f'phrase, sent: {kwp_lemma_list}, {sent_lemma_list[i:]}')
                    w = ' '.join(sent_lemma_list[i:i+len(kwp_lemma_list)])
                    if close_to(w, kwp_lemma):
                        # print(f'keyPHRASE match: {w}, {kwp_lemma}')
                        cats = kwp_df.loc[kwp_df.kwp_lemma == kwp_lemma].category.values.tolist()
                        # orig_w = ' '.join(parsed_sent_df.index[i:i+len(kwp_lemma_list)])
                        kwp_matches.loc[cats, 'kwpm'] = kwp_lemma
            elif close_to(sent_lemma_list[i], kwp_lemma):
                # print(f'keyWORD match: {sent_lemma_list[i]}, {kwp_lemma}')
                cats = kwp_df.loc[kwp_df['kwp_lemma'] == kwp_lemma].category.values.tolist()
                # orig_w = parsed_sent_df.index[i]
                kwp_matches.loc[cats, 'kwpm'] = kwp_lemma
    return kwp_matches


def find_pos_matches(parsed_df:pd.DataFrame) -> pd.DataFrame:
    """Get part of speech matches
    """
    pos_matches = pd.DataFrame(index=fb_cats, dtype=object)
    desc_regex = re.compile(r'(JJ.*)|(RB.*)')
    ly_regex = re.compile(r'.*ly')
    pos = parsed_df.loc[parsed_df.tag.apply(lambda x: bool(desc_regex.match(x))) == True]
    lys = parsed_df.loc[parsed_df.tag.apply(lambda x: bool(ly_regex.match(x))) == True]
    evals = pd.concat([pos,lys]).drop_duplicates()
    pos_matches.loc['evaluations', 'posm'] = ','.join(evals.loc[:,'lemma'])
    return pos_matches


def close_to(a:str, b:str, threshold:float=.4)->bool:
    """Tells if two words or phrases are close to each other
    """
    if a == b:
        score = 0
    else:
        al = a.split()
        bl = b.split()
        longer, shorter = (al, bl) if len(al)>=len(bl) else (bl,al)
        if all(elem in longer for elem in shorter):  # big contains small
            score = 0
        else:  # test closeness with derivationally related forms
            for i in range(len(al)):
                set_a = set([al[i]])
                set_a.add(lemma(al[i]))
                set_a.add(singularize(al[i]))
                set_a.update(lexeme(al[i]))
                al[i] = set_a      
            for i in range(len(bl)):
                set_b = set([bl[i]])
                set_b.add(lemma(bl[i]))
                set_b.add(singularize(bl[i]))
                set_b.update(lexeme(bl[i]))
                bl[i] = set_b
            if al == bl: score = 0
            else:
                total = ((len(al)+len(bl))/2)
                diffs = 0
                for i in longer:
                    if i not in shorter:
                        diffs+=1
                score = diffs/total
        # if score <= threshold: print(f'a, b, -> score: {al}, {bl} -> {score}')
    return score <= threshold


def feedback_to_english(fb_df:pd.DataFrame)->str:
    pass


def plot_cm(cm:np.ndarray, labels:list,
            normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", slice_size:int=1,
            norm_dec:int=2, plot_txt:bool=True, return_fig:bool=False, **kwargs)->Optional[plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
#         cm = confusion_matrix(slice_size=slice_size)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(labels)) # fucked it up?
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(labels)-.5,-.5)
                           
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        if not return_fig: return fig

# def most_confused(self, min_val:int=1, slice_size:int=1)->Collection[Tuple[str,str,int]]:
#     # This function is mainly copied from fastai src
#         "Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences."
#         cm = self.confusion_matrix(slice_size=slice_size)
#         np.fill_diagonal(cm, 0)
#         res = [(self.data.classes[i],self.data.classes[j],cm[i,j])
#                 for i,j in zip(*np.where(cm>=min_val))]
#         return sorted(res, key=itemgetter(2), reverse=True)