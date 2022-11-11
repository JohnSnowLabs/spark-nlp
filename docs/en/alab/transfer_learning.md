---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Training Parameters  
permalink: /docs/en/alab/transfer_learning
key: docs-training
modify_date: "2022-10-24"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

Annotation Lab supports the Transfer Learning feature offered by [Spark NLP for Healthcare 3.1.2](https://nlp.johnsnowlabs.com/docs/en/licensed_release_notes#support-for-fine-tuning-of-ner-models). 
This feature is available for project manages and project owners, but only if a valid Healthcare NLP license is loaded into the Annotation Lab. 

In this case, the feature can be activated for any project by navigating to the Train page. It requires the presence of a `base model` trained with [MedicalNERModel](https://nlp.johnsnowlabs.com/docs/en/licensed_release_notes#1-medicalnermodel-annotator).

If a MedicalNER model is available on the Models Hub section of the Annotation Lab, it can be chosen as a starting point of the training process. This means the `base model` will be Fine Tuned with the new training data.

When Fine Tuning is enabled, the same embeddings used for training the `base model` will be used to train the new model. Those need to be available on the Models Hub section as well. If present, embeddings will be automatically selected, otherwise users must go to the Models Hub page and download or upload them.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/transferLearning.gif" style="width:100%;"/>
