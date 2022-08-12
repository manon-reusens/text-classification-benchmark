#Import packages
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from time import time
import emoji 


class Preprocessor:

    def __init__(self,
                 is_tweet=False,
                 lemmatizer = WordNetLemmatizer(),
                 stopwords = stopwords.words('english'),
                 emojis = emoji.UNICODE_EMOJI_ENGLISH):
        '''
        The Preprocessor is used to convert raw text data  to a suitable format
        Args:
            is_tweet (bool) : If True, uses the TweetTokenizer to tokenize the text. If False, uses the standard word_tokenize method of nltk
            lemmatizer (object) : the desired nltk lemmatizer object
            stopwords (list) : a list of stopwords  to drop from the text
            emojis (list) : a list of emojis to drop from the text
        Output :
            corpus (list) : a list of processed text
        '''

        self.is_tweet = is_tweet
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords
        self.emojis = emojis
    
    def remove_unicode(self,text):
        #Remove unicode characters from text
        return text.encode('ascii','ignore').decode()

    def remove_punct(self,text):
        #Remove punctuation from string text
        punctuation = "!\"$%&'()*+,-?.../:;<=>[\]^_`{|}~#@"
        text = ''.join(char for char in text if char not in punctuation)
        return text

   
    def remove_repeated_letters(self, text):
        #If the same letters appears consecutively 3 times or more, reduce it to 2 occurences  (WIP implementation)
        text = re.compile(r'(.)\1{2,}', re.IGNORECASE).sub(r'\1'r'\1',text)
        return text

    def tokenize(self,text):
        #Tokenize the text
        if self.is_tweet:
            return TweetTokenizer().tokenize(text)
        else:
            return word_tokenize(text)

    def drop_digits(self,text):
        #Remove numbers from tokenized text
        text = [word for word in text if not word.isdigit()]
        return text

    def drop_emojis(self,text):
        #Remove (sequence of) emojis from tokenized text
        text = [word for word in text if word[0] not in self.emojis] 
        #If there is more than one emoji in the word token, just check the first one
        return text
        
    def drop_stopwords(self,text):
        #Remove stopwords from tokenized text
        text = [word for word in text if word not in self.stopwords]
        return text

    def drop_urls(self,text):
        #Remove urls from tokenized text
        text = [word for word in text if 'http' not in word]
        return text 
    
    def lemmatize(self,text):
        #Lemmatize text 
        text = [self.lemmatizer.lemmatize(word) for word in text]
        return text

    def preprocess(self,df):
        #Full preprocessing pipeline
        #Takes a dataframe as input
        start_time = time()
        corpus = df['text'].fillna('').str.lower().to_list()
        corpus = [self.remove_unicode(text) for text in corpus]
        corpus = [self.remove_punct(text) for text in corpus]
        corpus = [self.remove_repeated_letters(text) for text in corpus]
        corpus = [self.tokenize(text) for text in corpus]
        corpus = [self.drop_digits(text) for text in corpus]
        corpus = [self.drop_stopwords(text) for text in corpus]
        corpus = [self.drop_emojis(text) for text in corpus]
        corpus = [self.drop_urls(text) for text in corpus]
        corpus = [self.lemmatize(text) for text in corpus]
        corpus = [' '.join(text) for text in corpus]
        
        print('%s rows preprocessed in %s seconds'%(df.shape[0],time()-start_time))
        return corpus
