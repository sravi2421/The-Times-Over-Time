import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


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