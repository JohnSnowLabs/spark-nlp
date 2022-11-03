---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Models Hub
permalink: /docs/en/alab/models_hub
key: docs-training
modify_date: "2022-11-01"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

Annotation Lab offers tight integration with [NLP Models Hub](https://nlp.johnsnowlabs.com/models). Any compatible model and embeddings can be downloaded and made available to the Annotation Lab users for pre-annotations either from within the application or via manual upload.

NLP Models HUB page is accessible from the left navigation panel by users in the _UserAdmins_ group.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/models_hub.png" style="width:100%;" />

The Models Hub page lists all the pre-trained models and embeddings from NLP Models Hub that are compatible with the Spark NLP version present in the Annotation Lab. 

## Search

Search features are offered to help users identify the models they need based on their names. Additional information such as Library Edition, task for which the model was build as well as publication date are also available on the model tile. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/search.png" style="width:100%;"/>

Language of the model/embeddings is also available as well as a direct link to the model description page on the NLP Models Hub where you can get more details about the model and usage examples. 

## Filter

To make searching models/embeddings more efficient, Annotation Lab offers a Language filter. Users can select models/embeddings on the Models Hub page according to their language preference.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/model_language.png" style="width:100%;"/>


## Download 

By selecting one or multiple models from the list, users can download those to the Annotation Lab. The licensed (Healthcare, Visual, Finance or Legal) models and embeddings are available to download only when a valid license is present.

One restriction on models download/upload is related to the available disk space. Any model download requires that the double of its size is available on the local storage. If enough space is not available then the download cannot proceed.  

Disk usage view, search, and filter features are available on the upper section of the Models Hub page.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/storage.png" style="width:50%;" />


## Benchmarking

For the licensed models, benchmarking information is available on the Models Hub page. To check this click on the icon on the lower right side of the model tile. The benchmarking information can be used to guide the selection of the model you include in your project configuration. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/benchmarking.png" style="width:100%;"/>