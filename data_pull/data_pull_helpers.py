import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text, pos_tags_interest = ['ADJ', 'ADV']):
    '''
    preprocessing contains three steps:
    1) remove stop words
    2) lemmatize words
    3) add POS tag to any word which is a descriptor
    '''
    article = nlp(text)
    final_article = []
    for token in article:
        #remove stop word
        if token.is_stop:
            pass
        elif token.pos_=='SPACE':
            pass
        else:
            #tag POS of interest
            if token.pos_ in pos_tags_interest:
                #standardize the case of words
                if token.pos_ !='PROPN':
                    final_article.append(f'{token.lemma_.lower()}_POS_{token.pos_}')
                else:
                    #for proper nouns capitalize first letter, the remaining should be lower case
                    final_article.append(f'{token.lemma_[0]+token.lemma_[1:].lower()}_POS_{token.pos_}')
                    
            else:
                if token.pos_ !='PROPN':
                    final_article.append(token.lemma_.lower())
                else:
                    #for proper nouns capitalize first letter, the remaining should be lower case
                    final_article.append(token.lemma_[0]+token.lemma_[1:].lower())
    return ' '.join(final_article)