# Patent Evaluator 
# -- Increase Return of Patent Investments

The project is developed to predict potential usefulness of patents based on patent information and content, aiming to increase the overall return of patent investment for companies.

Patents serve as an important asset for technology companies. However, reports estimated that most patents failed to generate enough return to cover the investment <a href="http://ei.com/wp-content/uploads/downloadables/EI_Patent_Study_Singer.pdf" class="text">[1]</a>. Even though evaluating the potential of a patent based on its content is a difficult task even for experts, machine learning methods are able to help on this task by processing complex information and picking up subtle signals. The final model is estimated to generate about $2000 increased net value per applied patent.

## The web app is hosted at <a href="jennyyu.tech/patent" class="text">jennyyu.tech/patent</a>

## Files

## Data
University patents approved by US patent office were scraped from <a href="https://patents.google.com/" class="text">Google Patents</a>. The classifier model was trained on 10,030 patents in 2004-2007. In addition, a Word2Vec model was trained on 23,000 university patents in 2004-2010 to process the text data. The model was tested on 2000 patents in the held-out testing set. 

## Features
Patents are classified based on both text and non-text features. 
The text features were processed by a Word2Vec model trained on patent claims. 
The non-text features include 
- classification
- number of applications
- number of claims
- number of cited patents
- number of cited non-patent publications
- number of authors
- number of similar documents

## Models
### Word2Vec Model
A Word2Vec model was trained on the claims of 23,000 patents. The model was used to process each word in the claims of the training patents. The average vector of each patent serves as features in the random forest classifier.

### Random Forest classifier
Both the text features and non-text information were fed into a random forest model. Hyperparameters were tuned by 5-fold cross validation. The final model was selected based on a self-define metric described below. 

### The Metric
Models were selected by net value of investment during cross validation. The net value was calculated based on the model prediction, and estimated cost and value of patents. Here are the assumptions made for the calculation:
    
- Only patents predicted to be useful will be applied
- Only patents that are actually useful will generate value
- Cost of preparing and applying for a patent is $20,000 <a href="http://www.insidecounsel.com/2016/09/16/whats-a-patent-worth" class="text">[2]</a>
- Value of a “useful” patent is estimated to be around $30,000 <a href="http://www.tynax.com/transactions_patent_sale_guide.php#5-Valuation" class="text">[3]</a>


## Results
### Feature importance
The final model heavily relies on the text features to classify the patents, suggesting that patent claims contain essential information related to patent "usefulness". Moreover, some non-text features such as the number of applications and usage claims also play important roles. The result also suggests that the field of application (classification) and number of authors of the patent have little effect on the classification.

 <table align = "center" style="width:60em">
          <tr>
            <th>Group</th>
            <th>Features</th> 
            <th>Total importance</th>
          </tr>
          <tr>
            <td>Text features</td>
            <td>100 vector elements</td> 
            <td>0.904</td>
          </tr>
          <tr>
            <td>Top non-text features</td>
            <td># claims, # patent/non-patent references, #applications, # similar documents</td> 
            <td>0.081</td>
          </tr>
          <tr>
            <td>Number of authors</td>
            <td># authors</td> 
            <td>0.008</td>
          </tr>
          <tr>
            <td>Classification</td>
            <td>6 contrast-encoded features</td> 
            <td>0.007</td>
          </tr>
        </table>    
        
  ### Model performance
  On the 2000 patent from 2004-2007 in the held-out testing data, investment based on the model prediction is estimated to generate $2.79M increased net value compared to the baseline situation of applying for all the patents. The average return per applied patent is estimate to be increased by $2064 by the model.
