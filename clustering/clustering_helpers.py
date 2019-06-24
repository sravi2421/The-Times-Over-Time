import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import defaultdict



def get_topic_cluster_mapping(nmf_obj, tfidf_vectorizer, keywords, 
                              threshold = 5, n_top_words = 30):
    '''
    retrieves a mapping of clusters to topics based on provided keywords
    Args:
        nmf_obj (sklearn nmf_obj):      series containing text of article
        tdif_vectorizer (sklearn obj):  sklearn TdifVectorizer fit on current data
        keywords (dict):                dictionary of the form {'topic':[keyword1, keyword2...]}
        threshold (int):                minimum number of keyword/top topic words overlap 
                                        to assign to a topic
        n_top_words (int):              number of top words to compare against the keywords topic lists
    Returns:
        dict of the following format:   {0:topic_0, 1:topic_1, 2:'NAP'}
    '''
    topic_key = {}
    features = tfidf_vectorizer.get_feature_names()
    for i, topic in enumerate(nmf_obj.components_):
        topic_score_pairs = list(zip(nmf_obj.components_[i,:], features))
        topic_score_pairs = sorted(topic_score_pairs, key = lambda x: x[0], reverse = True)[:n_top_words]
        topic_scores = {**{key:0 for key in keywords}, **{'NAP':threshold-1}}
        for key in keywords.keys():
            overlap = len(list(set([x[1] for x in topic_score_pairs]) & set(keywords[key])))
            if overlap>threshold:
                topic_scores[key] = overlap
        scores_topics = {v: k for k, v in topic_scores.items()}
        top_score = sorted(scores_topics.keys(), reverse=True)[0]
        topic_key[i] = scores_topics[top_score]
    return topic_key

def update_default_dict(default_dict, elem_dict):
    '''
    Args:
        default_dict (collections.defaultdict):     dict to be updated
        elem_dict (dictionary):                     elements to be added
    return:
        default_dict (collections.defaultdict)
    '''
    for i in elem_dict:
        for word in elem_dict[i]:
            default_dict[word] += 1
    return default_dict

def print_dict_items(dict_):
    for i in dict_:
        print(f'topic{i}')
        print(' '.join(dict_[i]))
        
        
def get_top_words(nmf_obj, tfidf_vectorizer, n_top_words):
    '''
    Args:
        nmf_obj (sklearn nmf_obj):      series containing text of article
        tdif_vectorizer (sklearn obj):  sklearn TdifVectorizer fit on current data
        n_top_words (int):              number of top words to print
    Returns:
        topic_words_key (dict): dict of topic key and top words
    
    '''
    features = tfidf_vectorizer.get_feature_names()
    topic_words_key = {}
    for i, topic in enumerate(nmf_obj.components_):
        topic_score_pairs = list(zip(nmf_obj.components_[i,:], features))
        topic_words_key[i] = sorted(topic_score_pairs, key = lambda x: x[0], reverse = True)[:n_top_words]
    return topic_words_key

        
def gen_nmf_tfidf_model(text_series, n_topics = 5, 
                       tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=2,
                                                          max_features= 1000)):
    '''
    generates an nmf cluster and tfidf vectorizer object for a text series
    Args:
        text_series (pd series):        series containing text of article
        n_topics (int):                 number of topics
        tdif_vectorizer (sklearn obj):  sklearn TdifVectorizer object, with hyper params set
    
    Returns:
        dict of the following format:
        {'nmf': sklearn NMF object,
        'tdif_vectorizer': tdif_vectorizer fit on the current data
        'article_topics_df': pandas dataframe with article text}
    '''
    tfidf = tfidf_vectorizer.fit_transform(text_series)
    nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
    topic_score = pd.DataFrame(nmf.fit_transform(tfidf))
    topic_score['categ'] = topic_score[[n for n in range(n_topics)]].idxmax(axis=1)
    topic_score['categ_score'] = topic_score[[n for n in range(n_topics)]].max(axis=1)
    article_topics = (pd.DataFrame(text_series)
                      .reset_index(drop=True)
                      .join(topic_score[['categ']]))
    return {'nmf':nmf, 'tfidf_vectorizer':tfidf_vectorizer, 'article_topic_df':article_topics}


def text_score_multiply(score_text_tuple, multiple):
    final_text = []
    for score, word in score_text_tuple:
        final_text.append((int(round(score*multiple,0))*(word+' '))[:-1])
    return ' '.join(final_text)

def article_clustering(df, intervals, n_topics=5):
    interval_outputs = {}
    final_article_df = pd.DataFrame()
    for interval in intervals:
        print(interval)
        #initialize articles df for analysis
        curr_df = df[df['year'].between(interval[0],interval[1],
                                        inclusive = True)]
        #initialize tfidf object to be used
        tfidf  = TfidfVectorizer(max_df=0.80, min_df=2, max_features= 2000)
        #generate and fit nmf, and tfidf objects, place each article in a cluster
        curr = gen_nmf_tfidf_model(curr_df['text_final'], tfidf_vectorizer=tfidf, n_topics=n_topics)
        #add year to 'articles' df
        curr['article_topic_df']['years'] = (
            np.full(
                (curr['article_topic_df'].shape[0],1),
                f'{interval[0]} - {interval[1]}'))
        curr['article_topic_df'].index = curr_df.index
        interval_outputs[interval] = curr
        final_article_df = (final_article_df
                            .append(curr['article_topic_df']))
    return interval_outputs, final_article_df

def top_topic_words(interval_outputs, intervals):
    '''
    returns a weighted version of each topic's top words for each interval
    '''
    final_topic_df = pd.DataFrame()
    for ii in intervals:
        curr = interval_outputs[ii]
        # get the top words for each cluster
        top_30words = get_top_words(nmf_obj=curr['nmf'], tfidf_vectorizer=curr['tfidf_vectorizer'],n_top_words=30)
        #create topic text weighted by topic-word match strength 
        topics = (
            pd.DataFrame(text_score_multiply(top_30words[key], 10) for key in top_30words)
            .rename(columns={0:'topic_words_weighted'}))
        #add year to 'topics' df
        topics['years'] = np.full((topics.shape[0],1),f'{ii[0]} - {ii[1]}')
        words_only = []
        for i in range(len(top_30words.keys())):
            words_only.append(' '.join([word for score, word in top_30words[i]]))
        topics['words_only'] = words_only
        topics = topics.reset_index().rename(columns={'index':'cluster'})
        #save all outputs to their appropriate data structures

        final_topic_df = (final_topic_df
                          .append(topics)
                          .reset_index(drop=True))
    return final_topic_df