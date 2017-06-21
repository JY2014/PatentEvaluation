from flask import render_template
from flask import request, redirect
from flaskexample import app
import pandas as pd
import pickle
import numpy as np
import patent_scraper as ps


@app.route('/patent')
def patent_input():
    return render_template("input.html")

@app.route('/input')
def patent_input2():
    # load the list of patent numbers
    random_number = pickle.load(open('models/patent_numbers.p', 'r'))
    # randomly select a number
    choice = np.random.choice(range(len(random_number)), 1)
    # find the patent number
    patent_number = random_number[choice[0]]

    return render_template("input2.html", random_number = patent_number)


@app.route('/manual_input')
def manual_input():
    return render_template("manual_input.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/fill=US7948209B2')
def fill_1():
    return render_template("fill_US7948209B2.html")


@app.route('/fill=US8278036B2')
def fill_2():
    return render_template("fill_US8278036B2.html")


@app.route('/error')
def error():
    return render_template("error.html")


@app.route('/output')
def patent_output():
  #pull patent number from user input
  patent_number = request.args.get('patent_number')
      
  # remove leading and trailing space
  patent_number = patent_number.strip()
  # remove the spaces in the string
  patent_number = patent_number.replace(' ', '')
  
  if patent_number is not None:
     
      ### scrape patent information from the Google patent page
      # url to the patent page
      url = 'https://patents.google.com/patent/' + patent_number + '/en'
      soup = ps.generate_soup(url)
      
      # check if a patent is international patent (different format)
      if patent_number[:2] == 'WO':
          WO_index = 1
      else:
          WO_index = 0
      
      # read the predictors
      try:
          predictors = ps.read_patent_info(soup, WO_index)
      except:
          return redirect("/error")
      
      # scale the predictors
      scaler = pickle.load(open("models/final_model_scaler.p", 'r'))
      predictors = scaler.transform(predictors)
      
      #input the model
      model = pickle.load(open("models/final_model.p", 'r'))
      
      # predict on the patent
      y_pred = model.predict_proba(predictors)
      
      # format the result
      patents = []
      
      patents.append(dict(patent_number=patent_number, title=ps.find_patent_title(soup), proba = y_pred[0]))
      
      if y_pred[0][0] > 0.5:
          the_result = 'not useful' 
      else:
          the_result = 'useful'
      return render_template("output.html", patents = patents, the_result = the_result)
  
  else:
      return redirect("/error")



@app.route('/output_manual')
def patent_output2():
  #pull information from user input
  title = request.args.get('title').strip()
  num_authors = request.args.get('num_authors')
  patent_class = request.args.get('patent_class')
  num_applications = request.args.get('num_applications')
  patent_citations = request.args.get('patent_citations')
  non_patent_citations = request.args.get('non_patent_citations')  
  num_claims = request.args.get('num_claims')  
  claims = request.args.get('claims')
  num_similar_doc = request.args.get('num_similar_doc')
  
  # format the non-text features
  patent_nontext = [num_applications, patent_citations, non_patent_citations,
                    num_claims, num_similar_doc, num_authors]
  patent_nontext = np.asarray(patent_nontext)
  patent_nontext = patent_nontext.reshape((1, 6))
  
  # one-hot encode the classes
  class_one_hot = np.zeros(7)
  classes = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
    
  for i in range(len(classes)):
      ind_class = classes[i]
      if patent_class == ind_class:
          class_one_hot[i] = 1
  class_one_hot = class_one_hot.reshape((1, 7))
  # combine with other features
  patent_nontext = np.concatenate([class_one_hot, patent_nontext], axis = 1)
  
  
  # format the claims
  patent_claims = ps.format_claims(claims)
  # combine the text vector with the non-text features
  predictors = np.concatenate([patent_nontext, patent_claims], axis = 1)
  
  # scale the predictors
  scaler = pickle.load(open("models/final_model_scaler.p", 'r'))
  predictors = scaler.transform(predictors)
  
  #input the model
  model = pickle.load(open("models/final_model.p", 'r'))
  
  # predict on the patent
  y_pred = model.predict_proba(predictors)
  
  # format the result
  patents = []
  patents.append(dict(title=title, proba = y_pred[0]))
  
  # turn the result into binary
  if y_pred[0][0] > 0.5:
      the_result = 'not useful' 
  else:
      the_result = 'useful'
  return render_template("output_manual.html", patents = patents, the_result = the_result)
