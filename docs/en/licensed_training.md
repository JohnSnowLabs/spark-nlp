---
layout: docs
header: true
title: Training
permalink: /docs/en/licensed_training
key: docs-training
modify_date: "2020-08-10"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

### Training Datasets
These are classes to load common datasets to train annotators for tasks such as
Relation Model, Assertion models and more.

{% include_relative licensed_training_entries/Annotation_tool_Json_reader.md %}


### Assertion 

Trains AssertionDL, a deep Learning based approach used to extract Assertion Status from extracted entities and text.

{% include_relative licensed_training_entries/AssertionDLApproach.md %}
{% include_relative licensed_training_entries/AssertionLogRegApproach.md %}

### Token Classification

These are annotators that can be trained to recognize named entities in text.

{% include_relative licensed_training_entries/MedicalNer.md %}

### Text  Classification
These are annotators that can be trained to classify text into different classes, such as sentiment.

{% include_relative licensed_training_entries/DocumentLogRegClassifierApproach.md %}
{% include_relative licensed_training_entries/GenericClassifier.md %}

### Relation Models

{% include_relative licensed_training_entries/RelationExtractionApproach.md %}

### Entity Resolution

Those models predict what are the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.).

{% include_relative licensed_training_entries/SentenceEntityResolver.md %}
{% include_relative licensed_training_entries/ChunkEntityResolver.md %}


