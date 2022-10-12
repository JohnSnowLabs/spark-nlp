---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.6.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_6_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

### 2.6.0

#### Overview

We are honored to announce that Spark NLP Enterprise 2.6.0 has been released.
The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases.
The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

<div class="h3-box" markdown="1">

#### Models

##### Pretrained Pipelines:

The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch.
Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:

```python
pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

pipeline.annotate('my string')
```

Pipeline descriptions:

- ```explain_clinical_doc_carp``` a pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

- ```explain_clinical_doc_era``` a pipeline with ner_clinical_events, assertion_dl and re_temporal_events_clinical. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

- ```recognize_entities_posology``` a pipeline with ner_posology. It will only extract medication entities.

More information and examples are available here: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb.

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (English)

RE models:

```
re_temporal_events_clinical
re_temporal_events_enriched_clinical
re_human_phenotype_gene_clinical
re_drug_drug_interaction_clinical
re_chemprot_clinical
```
NER models:

```
ner_human_phenotype_gene_clinical
ner_human_phenotype_go_clinical
ner_chemprot_clinical
```
More information and examples here:
https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (German)

The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

- ```German Clinical NER``` model for 19 clinical entities

- ```German Legal NER``` model for 19 legal entities

- ```German ICD-10GM```

More information and examples here:

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/15.German_Legal_Model.ipynb

</div><div class="h3-box" markdown="1">

#### Other Pretrained Models

We now have Named Entity Disambiguation model out of the box.

Disambiguation models map words of interest, such as names of persons, locations and companies, from an input text document to corresponding unique entities in a target Knowledge Base (KB).

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb

Due to ongoing requests about Clinical Entity Resolvers, we release a notebook to let you see how to train an entity resolver using an open source dataset based on Snomed.

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_5_5">Version 2.5.5</a>
    </li>
    <li>
        <strong>Version 2.6.0</strong>
    </li>
    <li>
        <a href="release_notes_2_6_2">Version 2.6.2</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
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
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li class="active"><a href="release_notes_2_6_0">2.6.0</a></li>
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