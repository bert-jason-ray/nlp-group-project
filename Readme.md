# Natural Language Processing Group Project
> Contributors:
> - Eriberto Contreras
> - Jason Turner
> - Ray Cerna
___

> Link to final slide and notebook
> - 📰 **Link to slide for general audience click here: [Slide](https://www.canva.com/design/DAFHd4qLmko/zgSiqq_c_Y4hQgINUzrI8g/edit?utm_content=DA[…]m_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**
> - 📗 **Link to final notebook click here: [MVP](https://github.com/bert-jason-ray/nlp-group-project/blob/main/group_final_notebook.ipynb)**

### Project Summary
<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

For this project, our team utlizes NLP (Natural Language Processing) a form of programming and machine learning techniques, to help understand and make use
of large amounts of text data. In this case, we have selected 100+ repositories from GitHub with the word 'MusicBot' in the title as of July 25, 2022. Music Bots 
allow users to listen to music while they play games, work, or chat with friends online. We attempt to predict the programming language of each of these repositories 
by web-scrapping their README file contents.
___ 

#### Project Objectives
> - Build a model that can predict what programming language a repository will be.
> - Produce a well-documented jupyter notebook that contains our analysis.
> - Generate a Google-Slide presentation suitable for a general audience.
___

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
___

#### Questions

> - What are the most common words in READMEs?
> - Does the length of the README vary by programming language?
> - Do different programming languages use a different number of unique words?
> - Are there any words that uniquely identify a programming language?
___

#### Audience
> - General audience without an understanding of the topic.
___

#### Project Deliverables
> - A well-documented Jupyter notebook that contains our analysis, and direct link to that notebook in github.
> - Slides suitable for a general audience that summarizes our findings. 
> - A well-labeled visualization in our slides, link to the slides. 
> - A five minute presentation.
___

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

> - Conclusion:  This data set had over 150 repositories. We fit several different models and found that the KNN model was the most accurate. It predicted the main programming language with an accuracy of 56.25% on lemmatized data. We noticed that it is not very accurate when predicting TypeScript but this is because there were only two readmes with this programming language on our test data. In       the train data set there were only six of these we definitely need to increase our acquire data but overall we were pretty accurate in predicting JavaScript or Python as the programming language.

> - Recommendations: More Github data is required. Scrape more data for a more accurate model.

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

##### 🛑 Planning
> - [x] Scroll through links of GitHub's trending repositories, most forked repositores, and most starred repositories and vote on a topic.
> - [x] Create README.md with data dictionary, project objectives and goals.
> - [x] ...
___

##### ➡️ Planning 🛑 Acquisition
> - Create a web scraper that will summon two seperate data frames at a time named `df1` and `df2`.
> - Joined dataframes them using `append()` function.
> - Pulled the list of queries using the `.full_name` collumn and grab the `.to_list()` function.
> - Added list to our [`Acquire.py`](https://github.com/bert-jason-ray/nlp-group-project/blob/main/acquire.py) and define our list of queries as `REPOS`.
___

##### ➡️ Plannning ➡️ Acquisition 🛑 Preparation
> - Store functions needed to prepare the data; make sure the module contains the necessary imports to run the code. The final functions (prepare.py) should do the following:
>   - Convert all characters to lowercase.
>   - Normalize any unicode characters.
>   - Tokenized data.
>   - Create stemmed and lemmatized versions of the cleaned data.
>   - Handle any missing values and drop rows with nulls.
>   - Return the top six programming languages found in the repositories.
>   - Handle erroneous data and/or outliers that need addressing.
>   - String replace the words 'bot' and 'musicbot' since they are already present in the repo title.
>   - Create any new features, if made for this project.
>   - Split data into train, validate, and test sets.
>   - Import the prepare functions from the prepare.py module and use it to prepare the data in the final notebook.
___

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation 🛑 Exploration
> - Answer key questions and figure out features that can be used in answering key questions.
> - Create visualizations that work toward discovering variable relationships.
> - Summarize conclusions, provide clear answers to specific questions, and summarize any takeaways/action plan from the work above.
___

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation ➡️ Exploration 🛑 Modeling

> - Discovered Baseline Accuracy of 43%
> - Created Models on Lemmatized data:
>   - Logistic Regression
>   - Decision Tree Classifier
>   - Random Forest Classifier
>   - K-Nearest Neighbors Classifier
___

##### ➡️ Planning ➡️ Acquisition ➡️ Preparation ➡️ Exploration ➡️ Modeling 🛑 Delivery
> - Summarize findings at the beginning like we would for an Executive Summary.
> - Walk team through the analysis we did to answer questions which lead to findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers we are analyzing as well as offer insights and recommendations based on findings.

<hr style="border-top: 10px groove red; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproducing Our Project

You will need all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download prepare functions
- [ ] Scrap notebooks
- [ ] Run the final report
