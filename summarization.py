#importing libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  


def dict_table(text_string):
   
    #removing stop words
    stpwrds = set(stopwords.words("english"))
    
    wrds = word_tokenize(text_string, language='english', preserve_line=False)
    
    #reducing words to their root form
    stem = PorterStemmer()
    
    #creating dictionary for the word frequency table
    freqtble = dict()
    for wd in wrds:
        wd = stem.stem(wd)
        if wd in stpwrds:
            continue
        if wd in freqtble:
            freqtble[wd] += 1
        else:
            freqtble[wd] = 1

    return freqtble


def sent_scores(sentences, freqtble):   

    #algorithm for scoring a sentence by its words
    sent_wt = dict()

    for sentence in sentences:
        sent_wrd_cnt = (len(word_tokenize(sentence)))
        sent_wrd_cnt_without_stpwrds = 0
        for word_weight in freqtble:
            if word_weight in sentence.lower():
                sent_wrd_cnt_without_stpwrds += 1
                if sentence[:7] in sent_wt:
                    sent_wt[sentence[:7]] += freqtble[word_weight]
                else:
                    sent_wt[sentence[:7]] = freqtble[word_weight]

        sent_wt[sentence[:7]] = sent_wt[sentence[:7]] / sent_wrd_cnt_without_stpwrds

       

    return sent_wt

def avg_score(sent_wt):
   
    #calculating the average score for the sentences
    sum_values = 0
    for entry in sent_wt:
        sum_values += sent_wt[entry]

    #getting sentence average value from source text
    average_score = (sum_values / len(sent_wt))

    return average_score

def articlesummary(sentences, sent_wt, threshold):
    counter = 0
    summary = ''

    for sentence in sentences:
        if sentence[:7] in sent_wt and sent_wt[sentence[:7]] >= (threshold):
            summary += " " + sentence
            counter += 1

    return summary

def summary(article):
    
    #creating a dictionary for the word frequency table
    freqtble = dict_table(article)

    #tokenizing the sentences
    sentences = sent_tokenize(article)

    #algorithm for scoring a sentence by its words
    sent_score = sent_scores(sentences, freqtble)

    #getting the threshold
    threshold = avg_score(sent_score)

    #producing the summary
    article_summary = articlesummary(sentences, sent_score, 1.2 * threshold)

    return article_summary

if __name__ == '__main__':
    #fetching the content from the URL
    data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
    
    article = data.read()
    
    #parsing the URL content and storing in a variable
    article = BeautifulSoup.BeautifulSoup(article,'html.parser')
    
    #returning <p> tags
    para = article.find_all('p')
    
    content = ''
    #looping through the paragraphs and adding them to the variable
    for p in para:  
        content += p.text
    results = summary(content)
    print(results)
