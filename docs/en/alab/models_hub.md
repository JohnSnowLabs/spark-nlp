---
layout: docs
comment: no
header: true
title: NLP Models Hub
permalink: /docs/en/alab/models_hub
key: docs-training
modify_date: "2021-06-23"
use_language_switcher: "Python-Scala"
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

## Available Embeddings Tab
This tab lists all embeddings available to the Annotation Lab together with information on their source and date of upload/download. Like models, any compatible embeddings can be downloaded from NLP Models Hub. By default, glove_100d embeddings are included in the deployment.




## Custom Models/Embeddings Upload

Custom NER models or embeddings can be uploaded using the Upload button present in the top right corner of the page. The labels predicted by the uploaded NER model need to be specified using the Model upload form.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/upload_models.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The models& embeddings to upload need to be Spark NLP compatible. 

All available models are listed in the Spark NLP Pipeline Config on the Setup Page of any project and are ready to be included in the Labeling Config for preannotation.