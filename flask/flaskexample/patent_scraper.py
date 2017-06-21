# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import urllib
import io
from bs4 import BeautifulSoup
from datetime import datetime
import gensim
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

##### A list of functions to scrape information from Google Patent page #####


### function to read a html page
# input: a url
# output: BeautifulSoup output
def generate_soup(url):
    # read the page
    page = urllib.urlopen(url).read()
    soup = BeautifulSoup(page, "lxml")
    
    return soup


### function to read liscenced information for a patent
# input: BeautifulSoup Output from Google Patent result page
# output: number of times fee was paid
def find_maintainance_years(soup):
    # find legal events
    legal_events = soup.find_all('tr', {'itemprop':'legalEvents'})
    
    # initiate status for no fee payments
    fee_payments = 0
    
    # check how many fee payments there are
    for event in legal_events:
        # read each legal event
        info = event.find('td', {'itemprop':'title'})
        
        for title in info:
            # if licensed
            if title.string == 'Fee payment':
                # change the licensing status
                fee_payments = fee_payments + 1
                
                break

    return fee_payments


### function to read title of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: title with patent number
def find_patent_title(soup):
    # read the title 
    title_content = soup.title.string
    
    # extract the main title
    # find the components
    elements = title_content.split('-')
    # find the length of the patent number and google tag
    len_1 = len(elements[0])
    len_2 = len(elements[-1])
    
    # remove the elements, just leave the main title
    title = title_content[len_1+2 : len(title_content)-len_2-2].strip()
    
    return title


### function to extract abstract of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: abstract of the patent
def find_patent_abstract(soup):
    try: 
        # find the head
        head_content = soup.head
        # read the abstract
        abstract = head_content.find("meta",  {"name":"description"})['content']
        abstract = abstract
    except:
        abstract = None

    return abstract



### function to find top classification of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: top classification of a patent (one letter)
def find_patent_class(soup):
    try:
        # find the classification code
        # read the first letter as the top class
        classification = soup.find("span", {"itemprop" : "Code"}).text[0]
    except:
        classification = "NA"
    
    return classification



### function to find number of fine classifications of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: number of final classifications (potential applications) of a patent
def find_patent_applications(soup):
    applications = soup.find_all("meta", {"itemprop": "Leaf"})
    return len(applications)


### function to find the original assignee of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: name of original assignee
def find_ori_assignee(soup):
    univ = soup.find("dd", {"itemprop": "assigneeOriginal"}).string

    return univ



### function to find citations of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: number of patent and non-patent citations
def find_citation_nums(soup):
    # initiate at zeros
    patent_citations = 0
    non_patent_citations = 0
    
    # the citation numbers are written in the section titles
    sections = soup.find_all('h2')

    for section in sections:
        content = section.string

        if content is not None:
            # find the matching sections and read the number
            words = content.split(" ")
            # for patent citations
            if words[0] == 'Patent' and len(words) > 2:
                patent_citations = int(words[2][1:len(words[2])-1])
            # for non-patent citations
            elif words[0] == 'Non-Patent' and len(words) > 2:
                non_patent_citations = int(words[2][1:len(words[2])-1])

    return patent_citations, non_patent_citations


### function to read the content of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: background and summary text together in one string
def read_patent_content(soup):
    description = []
    
    # find the corresponding sections
    text = soup.find("section", {"itemprop" : "description"})
    
    for header in text.find_all('heading'):
        # read all the paragraphs
        for elem in header.next_siblings:
            if elem.name == 'p':
                description.append(elem.get_text())

    # join the paragraphs into a string
    description = ' '.join(description)

    return description



### function to read the claims of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: number of claims and claim text
def read_patent_claims(soup):
    claims = soup.find("section", {'itemprop': 'claims'})
    # number of claims
    num_claims = int(claims.find("span", {'itemprop': 'count'}).string)
    
    # read claim content
    claim_content = []
    # find the corresponding sections
    contents = claims.find_all('div', {'class':'claim-text'})

    for content in contents:
        if content.text is not None:
            claim_content.append(content.text)
    # join the lines
    claim_content = ' '.join(claim_content)
    
    # replace the \n by space
    claim_content = claim_content.replace('\n', ' ')
    
    return num_claims, claim_content    



### function to find the number of similar documents as a patent
# the date of publication of the document should be before the patent submission date
# input: BeautifulSoup Output from Google Patent result page
# output: number of similar documents as the patent
def count_similar_documents(soup):
    # find the patent publication date
    submission = soup.find("meta", {"name" : "DC.date", "scheme" : "dateSubmitted"})
    submission_date = submission['content']
    # reformat the date
    submission_date = datetime.strptime(submission_date , '%Y-%m-%d')
    
    # count similar documents published before the submission date
    doc_num = 0
    # read each document
    similar_documents = soup.find_all("tr", {"itemprop": "similarDocuments"})
    # chech the publication date of each document
    for document in similar_documents:
        # read publication date
        publication_date = document.find("time", {"itemprop": "publicationDate"})
        publication_date = publication_date.string
        # reformat into date
        # a small number of publications do not have the same date format: ignore them
        try:
            publication_date = datetime.strptime(publication_date , '%Y-%m-%d')
        
            # count if the date is earlier than the submission date
            if publication_date < submission_date:
                doc_num += 1
        except:
            continue
    
    return doc_num


### function to find the number of inventors of the patent
# mainly for the new patents input into the webapp
def find_num_inventors(soup):
    inventors = soup.find_all("meta", {"name" : "DC.contributor", "scheme" : "inventor"})
    return len(inventors)


### function to compute average word-vector for a text
def dec_vec(model, text):
    # store the vector for each word
    vectors = []
    
    # compute on each word
    for j in range(len(text)):
        try:
            vectors.append(model.wv[text[j]])
        except:
            continue
    
    if not vectors:
        vectors_mean = np.zeros((1, 100))
    else:
        vectors_mean = np.nanmean(vectors, axis = 0)
        vectors_mean = vectors_mean.reshape((1, 100))
    
    # return vector mean
    return vectors_mean



### function to tokenize and clean a text
def tokenize_cleaning(text):
    # tokenize the text first
    try:
        tokens = word_tokenize(text.decode('utf-8'))
    except:
        tokens = word_tokenize(text)
    
    # lowercase all the words
    tokens = [w.lower() for w in tokens]
    
    # clean up stop words and punctuations 
    stop_list = stopwords.words('english') + list(string.punctuation)

    tokens_no_stop = [token for token in tokens
                        if token not in stop_list]            

    # use lemma instead
    # reason: remove the influence of plural or tense
    # but retain the subtle difference in legal writting
    lemmatizer = WordNetLemmatizer()
    tokens_lemma = [lemmatizer.lemmatize(token) for token in tokens_no_stop]
    
    # remove numbers (the actual values are not useful)
    tokens_no_num = []
    for token in tokens_lemma:
        try:
            float(token)
        except:
            tokens_no_num.append(token)
    
    return tokens_no_num



### function to format claims of a patent by trained word2vec
# for the new patents input into the webapp
# it returns a vector of size (1, 100)
def format_claims(text):
    # tokenize the text first
    cleaned_text = tokenize_cleaning(text)
    
    # read the model
    word2vec_model = Word2Vec.load('models/word2vec_claims_final')
    
    # compute the vector for the cleaned text
    vec = dec_vec(word2vec_model, cleaned_text)
    
    return vec


### function to read all the relevant information of a new patent
# intercept column is added to the features (first one)
def read_patent_info(soup, WO_index):
    
    # add top classification of the patent
    patent_class = find_patent_class(soup)
    # one-hot encode the classes
    class_one_hot = np.zeros(7)
    classes = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for i in range(len(classes)):
        ind_class = classes[i]
        if patent_class == ind_class:
            class_one_hot[i] = 1
    class_one_hot = class_one_hot.reshape((1, 7))
    
    # number of fine applications
    num_applications = find_patent_applications(soup)
    # number of citations
    patent_citations, non_patent_citations = find_citation_nums(soup)
    
    # add similar documents
    similar_doc_num = count_similar_documents(soup)
    
    # add claims
    if WO_index == 0:
        num_claims, claim_content = read_patent_claims(soup)
    else:
        num_claims, claim_content = read_patent_claims_WO(soup)
    
    # find number of inventors
    num_authors = find_num_inventors(soup)
    
    # put all the information together
    # all other features except for classification
    patent_nontext = [num_applications, patent_citations, non_patent_citations,
                      num_claims, similar_doc_num, num_authors]
    patent_nontext = np.asarray(patent_nontext)
    patent_nontext = patent_nontext.reshape((1, 6))
    # combine with classification
    patent_nontext = np.concatenate([class_one_hot, patent_nontext], axis = 1)
    
    # format claims by the trained word2vec
    patent_claims = format_claims(claim_content)
    # combine the text vector with the non-text features
    patent_features = np.concatenate([patent_nontext, patent_claims], axis = 1)
    
    return patent_features


### function to read claims for WO patents
# the format is different for international patents
def read_patent_claims_WO(soup):
    claims = soup.find("section", {'itemprop': 'claims'})
    # number of claims
    numbers = claims.find_all('claim')
    number = numbers[len(numbers)-1]
    num_claims = int(number['num'])
    
    # read claim content
    claim_content = []
    # find the corresponding sections
    contents = claims.find_all('div', {'class':'claim-text'})

    for content in contents:
        if content.string is not None:
            claim_content.append(content.string)
    # join the lines
    claim_content = ' '.join(claim_content)
    
    return num_claims, claim_content  