# GSurvWeb

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://woochan-hwang-gsurvweb-appmain-ke01om.streamlitapp.com/)
![Tests](https://github.com/woochan-hwang/GSurvWeb/actions/workflows/main.yml/badge.svg)

#### Open source interactive web application for transplant graft survival prediction  

## Project Scope
This project is focused on creating an interactive web application with tools specific for transplant graft survival prediction. The project started in 2021 with the motivation of analyzing a database of renal transplants within Guy's and St.Thomas' NHS Trust between 2009-2019. The current version is hence developed with certain assumptions that may only be specific to our database. We hope to create an application that covers most basic use cases for clinician researchers with limited software engineering experience and has the flexibility to allow more advanced users to extend custom models and functions with ease. Patient data is confidential and will remain excluded from this repository. 


### **Table of Contents**
<!-- created with help from https://github.com/ekalinin/github-markdown-toc and further manual adjustments -->

* [Project Scope](#project-scope)
* [How to start using the app](#how-to-start-using-the-app)
* [Deploy locally](#deploying-locally)
   * [Requirements](#requirements)
   * [Launch](#launch)
   * [Data Guidance](#data-guidance)
   * [Developer mode](#developer-mode)
   * [Custom Models](#custom-models)
   * [Unit testing](#unit-testing)
   * [Tips](#streamlit-tips)
* [Feedback](#feedback)
* [Contributors](#contributors)
* [Citing Us](#citing)
* [License](#license)

## How to start using the app

To launch the web app simply click on the streamlit app icon under the title. The web app itself is self explanatory in most cases, however you may refer to the hints next to each input widget for more details. We use [Streamlit Cloud](https://streamlit.io/cloud) for deployment at the time being and is therefore limited in computational availability (up to 1GB RAM). For large datasets, we recommend following the instructions [below](#deploy-locally) to host the web app locally on your local machine. Depending on the feedback and usage, we will consider migrating to other services (i.e. AWS etc.) to allow for more computational power. 

This app has been developed based on [Streamlit](https://streamlit.io/), a pythonic open-source app framework for Machine Learning. We would strongly recommend joining the streamlit community if you are used to handling data but less so in front-end development of web apps. 


## Deploy Locally

### Requirements

This project is set up using Docker. Use the provided Dockerfile and .devcontainer.json to build your docker image. 'Requirements.txt' contains all dependencies and will be automatically run when set up using devcontainer. 

Alternatively, you can set up your own python environment (ver>=3.8) and use pip.
 ```bash
pip install -r requirements.txt
```

### Launch 

To launch the streamlit web app on an local machine:
```bash
streamlit run main.py
```

### Data Guidance

As explained in the [project scope](#project-scope), the current release is focused on use cases based on a database collected from a single institution. Therefore there are some assumptions that have been made regarding the format of the data uploaded for analysis. Please use the [example template](https://github.com/woochan-hwang/GSurvWeb/blob/main/App/data/example_data_template.xlsx) provided as a guidance. 

Please note the additional sheets in the template file. The 'Data Code' sheet details what type of data each variable is. This allows the app to show appropriate options. The 'Data Range' sheet is specific for visualization of the cox proportional hazards model. Any continuous variable that will be used as an input for the cox model should be included in this sheet. 

### Developer Mode

For users extending this repository, we have provided command line options to help with the development process. The develop mode will allow you to specify a path to a local file rather than using the upload button within the web app. It will also show develop mode in the app mode selection box which where you can implement beta stage models. 

```bash
# custom arguments must be passed after two dashes, otherwise they will be interpreted as arguments to Streamlit itself.
streamlit run main.py -- -h  # show possible CLI arguments
streamlit run main.py -- -v  # verbose

# run app in developer mode to include beta functions and additional features
streamlit run main.py -- -d
streamlit run main.py -- --develop

# the following will be interpreted as a Streamlit option and give an error
streamlit run main.py --dev

# specify path to local data instead of using the upload option 
# only available in develop mode 
streamlit run main.py -- -d, -p='example/path_to/data.xlsx'
streamlit run main.py -- --develop, --path_to_data='example/path_to/data.xlsx'
```

### Custom Models

To create your own custom model and deploy it within the app, create a new model script under ```app/components/models/``` and create a class object inheriting ```python class BaseModel```. This will allow you to access all the data preprocessing and training hyperparameter settings. Inheriting ```python class BaseModel``` forces you to implement all necessary methods to run the web app smoothly. 

Once you have created a script for your custom script, update ```app/components/interactive.py``` to import your model and add appropriate parameter toggles. Once you are familiar with the streamlit interface, move onto ```app/components/experiment.py``` script to automate your experiments across various training parameters. 

```bash
# Structure of GSurvWeb
main.py
app/
├─ components/
│  ├─ models/
│  │  ├─ base_model.py
│  │  ├─ support_vector_machine_model.py
│  │  ├─ random_forest_model.py
│  ├─ data_summary.py
│  ├─ experiment.py
│  ├─ interactive.py
tests/
```

### Unit Testing

This project uses the [Pytest](https://docs.pytest.org/en/7.1.x/) framework for setting up unit tests. The current tests are set up to check deployment of mainly the frontend api rather than the analysis of the provided data. Testing of the model's performance given various possible user uploads are essential in making this project more reliable. The authors have only checked reliability for the data used for our publication. 

For users who introduce custom models, we would strongly recommend writing appropriate test scripts as well. 

To run tests before deployment, simply run: 

```bash
pytest  # use -v option for detailed test results
```

### Streamlit Tips

Streamlit can be deployed using streamlit cloud. However, one may wish to use other cloud services for development or deployment. In the case of the authors, we used the beta version of github codespaces to develop the initial version of this project. In such cases, the following script may circumvent some issues. [Click here](https://docs.streamlit.io/knowledge-base/deploy/remote-start) for more information on this issue.

```bash
streamlit run main.py --server.enableCORS=false --server.enableWebsocketCompression=false -- [custom arguments]
```


## Feedback 

We welcome all feedback including bug reports, pull requests, general remarks and questions. 

You may contact the authors via [email](mailto:woochan.hwang14@alumni.imperial.ac.uk).


## Contributors 

The work was started by a group of transplant surgeons at [Guy's and St.Thomas' NHS Foundation Trust](https://www.guysandstthomas.nhs.uk/) who were interested in learning about machine learning in clinical data. Dr Woochan Hwang joined as a junior doctor in the renal department. 

**Authors:**
Dr Woochan Hwang, Mr Usman Harron, Mr Ravindhran Bharadhwaj, Mr Toroth Ameen, Mr Pankaj Chandak, Miss Zakri Rhana 


## Citing 
This work is currently submitted for peer review. 


## License

Copyright holders Hwang 2022. 

Distributed under the [Apache License 2.0](https://github.com/woochan-hwang/GSurvWeb/blob/main/LICENSE). 
