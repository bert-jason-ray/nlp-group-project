## Natural Language Processing Group Project

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

For this project, our team utlizes NLP (Natural Language Processing) a form of programming and machine learning techniques, to help understand and make use
of large amounts of text data. In this case, we have selected 100+ repositories from GitHub with the word 'MusicBot' in the title as of July 25, 2022. Music Bots 
allow users to listen to music while they play games, work, or chat with friends online. We attempt to predict the programming language of each of these repositories 
by web-scrapping their README file contents. 

#### Project Objectives
> - Build a model that can predict what programming language a repository will be.
> - Produce a well-documented jupyter notebook that contains our analysis.
> - Generate a Google-Slide presentation suitable for a general audience.

#### Goals
> - Build a dataset from a list of 100+ GitHub repositories we have decided to scrape.
> - Generate the list of repositories programmaticly using web scraping techniques.
> - Explore and visualize the natural language data we have acquired.
> - Transform the documents into a form that can be used in a machine learning model.
> - Use the programming language of the repository as the label to predict.
> - Fit several different models.
> - Use several different representations of the text.
> - Build a function that takes in the README file and tries to predict the programming language.
> - Document process well enough to be presented or read like a report.

#### Audience
> - General audience without an understanding of the topic.

#### Project Deliverables
> - A well-documented Jupyter notebook that contains our analysis, and direct link to that notebook in github.
> - Slides suitable for a general audience that summarizes our findings. 
> - A well-labeled visualization in our slides, link to the slides. 
> - A five minute presentation.


#### Data Dictionary


| Column            | Non-Null Count Datatype         | Definition                                              |
|-------------------|---------------------------------|---------------------------------------------------------|
| repo              | 160 non-null: object            | The title of the repository                             |
| language          | 160 non-null: object            | The programming language used in the repository         |
| readme_contents   | 160 non-null: object            | The original README contents of the repository          |
| clean             | 160 non-null: object            | The cleaned version of the README                       |
| stemmed           | 160 non-null: object            | The cleaned, stemmed version of the readme              |
| lemmatized        | 160 non-null: object            | The cleaned, lemmatized version of the readme           |
| label             | 160 non-null: object            | The programming language label; the target variable     |

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

> - Question: 
> - Actions: 
> - Conclusions:  
> - Recommendations: 

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

##### 🛑 Planning
- [x] Create README.md with data dictionary, project objectives and goals.
- [x] ...
- [x] ...

##### ➡️ Planning 🛑 Acquisition
> - Store functions that are needed to acquire data from the database server; make sure the acquire.py module contains the necessary imports for anyone with database access to run the code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the final notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, etc.).
> - Plot distributions of individual variables.
___

##### ➡️ Plannning ➡️ Acquisition 🛑 Preparation
> - Store functions needed to wrangle the data; make sure the module contains the necessary imports to run the code. The final functions (wrangle.py) should do the following:
    - Since there is no modeling to be done for this project, there is no need to split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare functions from the wrangle.py module and use it to prepare the data in the final notebook.
___

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation 🛑 Exploration
> - Answer key questions, our hypotheses, and figure out the features that can be used in answering key questions.
> - Create visualizations that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify anomalies in curriculum logs, identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize conclusions, provide clear answers to specific questions, and summarize any takeaways/action plan from the work above.

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation ➡️ Exploration 🛑 Modeling
> - This project does not contain any modeling.
___

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation ➡️ Exploration ➡️ Modeling 🛑 Delivery
> - Summarize findings at the beginning like we would for an Executive Summary.
> - Walk team through the analysis we did to answer questions which lead to findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers we are analyzing as well as offer insights and recommendations based on findings.

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproducing Our Project

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

You will need all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Access to CodeUp MySql server
- [ ] Download [[wrangle functions]]
- [ ] Scrap notebooks
- [ ] Run the final report
