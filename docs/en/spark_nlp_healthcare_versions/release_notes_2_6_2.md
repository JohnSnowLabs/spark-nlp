---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.6.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_6_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

### 2.6.2

#### Overview
We are very happy to announce that version 2.6.2 of Spark NLP Enterprise is ready to be installed and used.
We are making available Named Entity Recognition, Sentence Classification and Entity Resolution models to analyze Adverse Drug Events in natural language text from clinical domains.

#### Models

##### NERs
We are pleased to announce that we have a brand new named entity recognition (NER) model for Adverse Drug Events (ADE) to extract ADE and DRUG entities from a given text.

ADE NER will have four versions in the library, trained with different size of word embeddings:

`ner_ade_bioert` (768d Bert embeddings)  
`ner_ade_clinicalbert` (768d Bert embeddings)  
`ner_ade_clinical` (200d clinical embeddings)  
`ner_ade_healthcare` (100d healthcare embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

We are also releasing our first clinical pretrained classifier for ADE classification tasks. This new ADE classifier is trained on various ADE datasets, including the mentions in tweets to represent the daily life conversations as well. So it works well on the texts coming from academic context, social media and clinical notes. It’s trained with `Clinical Biobert` embeddings, which is the most powerful contextual language model in the clinical domain out there.

##### Classifiers
ADE classifier will have two versions in the library, trained with different Bert embeddings:

`classifierdl_ade_bioert` (768d BioBert embeddings)  
`classifierdl_adee_clinicalbert` (768d ClinicalBert embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

##### Pipeline
By combining ADE NER and Classifier, we are releasing a new pretrained clinical pipeline for ADE tasks to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:
```python
pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

pipeline.annotate('my string')
```
`explain_clinical_doc_ade` is bundled with `ner_ade_clinicalBert`, and `classifierdl_ade_clinicalBert`. It can extract ADE and DRUG clinical entities, and then assign ADE status to a text (`True` means ADE, `False` means not related to ADE).

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)


##### Entity Resolver
We are releasing the first Entity Resolver for `Athena` (Automated Terminology Harmonization, Extraction and Normalization for Analytics, http://athena.ohdsi.org/) to extract concept ids via standardized medical vocabularies. For now, it only supports `conditions` section and can be used to map the clinical conditions with the corresponding standard terminology and then get the concept ids to store them in various database schemas.
It is named as `chunkresolve_athena_conditions_healthcare`.

We added slim versions of several clinical NER models that are trained with 100d healthcare word embeddings, which is lighter and smaller in size.

`ner_healthcare`
`assertion_dl_healthcare`
`ner_posology_healthcare`
`ner_events_healthcare`

##### Graph Builder
Spark NLP Licensed version has several DL based annotators (modules) such as NerDL, AssertionDL, RelationExtraction and GenericClassifier, and they are all based on Tensorflow (tf) with custom graphs. In order to make the creating and customizing the tf graphs for these models easier for our licensed users, we added a graph builder to the Python side of the library. Now you can customize your graphs and use them in the respected models while training a new DL model.

```python
from sparknlp_jsl.training import tf_graph

tf_graph.build("relation_extraction",build_params={"input_dim": 6000, "output_dim": 3, 'batch_norm':1, "hidden_layers": [300, 200], "hidden_act": "relu", 'hidden_act_l2':1}, model_location=".", model_filename="re_with_BN")
```
More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_6_1">Version 2.6.1</a>
    </li>
    <li>
        <strong>Version 2.6.2</strong>
    </li>
    <li>
        <a href="release_notes_2_7_0">Version 2.7.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li class="active"><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>