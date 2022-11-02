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

The Models Hub page lists all the pre-trained models and embeddings from NLP Models Hub that are compatible with the Spark NLP version present in the Annotation Lab. By selecting one or multiple models from the list, users can download those to the Annotation Lab. The licensed (Healthcare, Visual, Finance or Legal) models and embeddings are available to download only when a valid license is present.

One restriction on models download/upload is related to the available disk space. Any model download requires that the double of its size is available on the local storage. If enough space is not available then the download cannot proceed.  

Disk usage view, search, and filter features are available on the upper section of the Models Hub page.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/storage.png" style="width:50%;" />

Since version 2.8.0, users can view the full version specification and the language of the model/embeddings on the Models Hub page of the model. 

To make searching models/embeddings more efficient, Annotation Lab offers a Language filter. Users can select models/embeddings on the Models Hub page according to their language preference.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/model_language.png" style="width:100%;"/>
