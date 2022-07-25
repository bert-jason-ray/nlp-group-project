## Natural Language Processing Group Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

For this project, our team utlizes NLP (Natural Language Processing) a form of programming and machine learning techniques, to help understand and make use
of large amounts of text data. In this case, we have selected 100+ repositories from GitHub with the word 'MusicBot' in the title as of July 25, 2022. Music Bots 
allow users to listen to music while they play games, work, or chat with friends online. We attempt to predict the programming language of each of these repositories 
by web-scrapping their README file contents. 

#### Project Objectives
> - Build a model that can predict what programming language a repository will be.
> - Produce a well-documented jupyter notebook that contains our analysis.
> - 

#### Goals
> - Answer questions for CodeUp staff by analyzing data from curriculum_logs.
> - Prepare a single slide that summarizes most important points which will be incorporated into an existing presentation.
> - Document process well enough to be presented or read like a report.

#### Audience
> - CodeUp Board Members
> - CodeUp Students!

#### Project Deliverables
> - A final report notebook 
> - All necessary modules to make project reproducible
> - An email before the due date which includes:
>    - answering all questions (details are clearly communicated in email for leader to convey/understand)
>    - a link to final notebook
>    - an executive summary google slide in form to present


#### Data Dictionary
- Note: Includes only those features selected for full EDA and Modeling:

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| ? | 2820 non-null: object | Earthlike or Not-Earthlike gravity measurement, based on planet's radius |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| ?      | 2820 non-null: int64 | number of planets in system |



#### Initial Hypotheses

> - **Hypothesis 1 -**
> - H1

> - **Hypothesis 2 -** 
> - H2 (additional if needed)

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Question: 
> - Actions: 
> - Conclusions:  
> - Recommendations: 

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- [x] Create README.md with data dictionary, project objectives and goals, come up with initial hypotheses.
- [x] ...
- [x] ...

___

##### Plan -> Acquire
> - Store functions that are needed to acquire data from the database server; make sure the acquire.py module contains the necessary imports for anyone with database access to run the code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the final notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, etc.).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare/Wrange
> - Store functions needed to wrangle the data; make sure the module contains the necessary imports to run the code. The final functions (wrangle.py) should do the following:
    - Since there is no modeling to be done for this project, there is no need to split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare functions from the wrangle.py module and use it to prepare the data in the final notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, our hypotheses, and figure out the features that can be used in answering key questions.
> - Create visualizations that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify anomalies in curriculum logs, identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize conclusions, provide clear answers to specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - This project does not contain any modeling.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Summarize findings at the beginning like we would for an Executive Summary.
> - Walk team through the analysis we did to answer questions which lead to findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers we are analyzing as well as offer insights and recommendations based on findings.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce Our Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Access to CodeUp MySql server
- [ ] Download [[wrangle functions]]
- [ ] Scrap notebooks
- [ ] Run the final report

##### Credit to Faith Kane (https://github.com/faithkane3) for the format of this README.md file.