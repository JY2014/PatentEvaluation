import numpy as np
import pandas as pd
import urllib
import io
from bs4 import BeautifulSoup
from datetime import datetime

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
# output: title and patent number of the patent
def find_patent_title(soup):
    # read the title and split by '-'
    title_content = soup.title.string.split("-")
    
    # extract the body of the title
    title = title_content[1]
    # remove the space
    title = title[1:len(title)-3]
    # remove '\n'
    title = title.split("\n")[0]
    
    # read the patent number
    patent_num = title_content[0]
    # remove space
    patent_num = patent_num[:len(patent_num)-1]
    
    return title, patent_num



### function to extract abstract of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: abstract of the patent
def find_patent_abstract(soup):
    # find the head
    head_content = soup.head
    # read the abstract
    abstract = head_content.find("meta",  {"name":"description"})['content']

    return abstract



### function to find top classification of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: top classification of a patent (one letter)
def find_patent_class(soup):
    # find the top class
    classification = soup.find_all("span", {"itemprop" : "Code"})[1]
    return classification.text



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
# output: background and summary text
def read_patent_content(soup):
    background = []
    summary = []
    
    # find the corresponding sections
    text = soup.find("section", {"itemprop" : "description"})
    
    for header in text.find_all('heading'):
        if header.get_text() == "BACKGROUND OF THE INVENTION":
            # read the content
            for elem in header.next_siblings:
                if elem.name == 'heading':
                    break
                if elem.name == 'p':
                    background.append(elem.get_text())

        if header.get_text() == "SUMMARY OF THE INVENTION":
            for elem in header.next_siblings:
                if elem.name == 'heading':
                    break
                if elem.name == 'p':
                    summary.append(elem.get_text())
    
    # join the paragraphs into a string
    background = ' '.join(background)
    summary = ' '.join(summary)
    
    return background, summary



### function to read the claims of a patent
# input: BeautifulSoup Output from Google Patent result page
# output: number of claims and claim text
def read_patent_claims(soup):
    claims = soup.find("section", {'itemprop': 'claims'})
    # number of claims
    num_claims = len(claims)
    
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
