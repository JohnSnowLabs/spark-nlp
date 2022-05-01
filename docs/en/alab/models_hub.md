---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: NLP Models Hub
permalink: /docs/en/alab/models_hub
key: docs-training
modify_date: "2021-06-23"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

The Annotation Lab 1.8.0 offers a tight integration with [NLP Models Hub](https://nlp.johnsnowlabs.com/models). Any compatible NER model and Embeddings can be downloaded and made available to the Annotation Lab users for preannotations either from within the application or via manual upload. 

Models Hub page can be accessed via the left navigation menu by users in the UserAdmins group.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/models_hub.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


The Models Hub page has three tabs that hold information about models and embeddings.

## Models Hub Tab
This tab lists all pretrained NER models and embeddings from NLP Models Hub which are compatible with Spark NLP 2.7.5 and which are defined for English language. By selecting one or multiple items from the grid view, users can download them to the Annotation Lab. The licensed/Healthcare models and embeddings are available to download only when a valid license is uploaded.

One restriction that we impose on models download/upload relates to the available disk space. Any model download requires that the double of its size is available on the local storage. If not enough space is available the download cannot proceed.  

Disk usage view, search, and filter features are available on the upper section of the Models Hub page.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/storage.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Available Models Tab
All the models available in the Annotation Lab are listed in this tab. The models are either trained within the Annotation Lab, uploaded to Annotation Lab by admin users, or downloaded from NLP Models Hub. General information about the models like labels/categories and the source (downloaded or trained or uploaded) of the model can be viewed. It is possible to delete any model or redownload failed ones by using the overflow menu on the top right corner of each model.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/re_download.png" style="width:60%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Since release 2.8.0 users can view the full version specification and the language of the model/embeddings on the Models Hub page of the model.  Also to make searching of models/embeddings more efficient Annotation Lab offers a Language filter. This can be used to select models/embeddings on the Models Hub page according to their language.

 ![untitled (2)](https://user-images.githubusercontent.com/73094423/158778100-d6620491-a411-42e4-87cf-2e81606011f3.png)

## Download of model dependencies

In previous versions of Annotation Lab, when a user downloaded a model from the Models Hub page, for example, ner_healthcare_de, and tried to reuse labels from this model into a new project, via the Predefined labels page, an error message was preventing the saving of the configuration. The error (e.g. "w2v_cc_300d is not present in the machine") appeared because the embeddings used to train the model were not available to the Annotation Lab server. Starting with version 2.8.0, Annotation Lab automatically downloads all necessary dependencies along with the model. So, the user does not have to manually handle this step – e.g. navigate to the Models Hub page and download the dependent embeddings.
 
 ![Screen Recording 2022-03-16 at 9 58 12 PM](https://user-images.githubusercontent.com/17021686/158637378-9c01b3f3-6ba4-4bcb-8d34-8eb3141df484.gif)

## Available Embeddings Tab
This tab lists all embeddings available to the Annotation Lab together with information on their source and date of upload/download. Like models, any compatible embeddings can be downloaded from NLP Models Hub. By default, glove_100d embeddings are included in the deployment.


## Available Rules Tab

Spark NLP for Healthcare supports rule-based annotations via the ContextualParser Annotator. In this release, Annotationlab adds support for creating and using ContextualParser rules in NER project. 

Any user with admin privileges can see and edit the available rules under the `Available Rules` tab on the `Models Hub` page. Users can create new rules using the `+ Add Rules` button.

<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/rules_tab.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

There are two types of rules supported:

- **`Regex Based:`** Users can define a regex that will be used to label all possible hit chunks and label them as being the target entity. For example, for labeling height entities the following regex can be used `[0-7]'((0?[0-9])|(1(0|1)))`. All hits found in the task text that match the regex, are pre-annotated as heights.

- **`Dictionary Based:`** Users can define and upload a CSV dictionary of keywords that cover the list of chunks that should be annotated as a target entity. For example, for the label female: woman, lady, girl, all occurrences of stings woman, lady, and girl within the text of a given task will be perannotated as female.   

<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/types_of_rules.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

After adding a rule on `Models Hub` page, the `Project Owner` or `Manager` can add the rule to the configuration of the project where he wants to use it. This can be done via the `Rules` tab from the `Project Setup` page under the `Project Configuration` tab. A valid Spark NLP for Healthcare license is required to deploy rules from project config.


The user is notified when a rule was edited via an alert message "Redeploy preannotation server to apply these changes" on the rule edit form so that the users can redeploy the preannotation model. 
 
 ![redeploy-rules](https://user-images.githubusercontent.com/17021686/158801947-9cd847b7-abdf-42e9-b621-b406ad62b826.png)



## Import and Export Rules

From this version, Annotationlab provides the feasibility of importing and exporting Contextual parser Rules from the Model Hub page.

**Import Rules**
 
This release provides user's with the functionality to import rules, now users can import these rules from the Models Hub page under the Rules tab. Users can import both dictionaries as well as regex rules. Rules can be imported in the following formats:-
 1. A single **JSON** file.
 2. A **ZIP** archive containing multiple individual rules.
 
 ![RulesImport](https://user-images.githubusercontent.com/17021686/158798253-d2334cd2-96f5-440e-921b-bcd60cd3d709.gif)

**Export Rules**

To export any rule the user needs to navigate to the Models Hub page under the Rules tab, then select from the available rules and click on the Export button. Rules will be downloaded in ZIP format containing files in JSON format. Also, these exported rules can be imported into AnnotationLab.
 
 ![RulesExport](https://user-images.githubusercontent.com/17021686/158798831-4138bd1c-82f1-4624-8f16-0c13833a981e.gif)

## Custom Models/Embeddings Upload

Custom NER models or embeddings can be uploaded using the Upload button present in the top right corner of the page. The labels predicted by the uploaded NER model need to be specified using the Model upload form.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/upload_models.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The models& embeddings to upload need to be Spark NLP compatible. 

All available models are listed in the Spark NLP Pipeline Config on the Setup Page of any project and are ready to be included in the Labeling Config for preannotation.