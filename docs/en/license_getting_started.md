---
layout: docs
header: true
title: Spark NLP for Healthcare
permalink: /docs/en/license_getting_started
key: docs-licensed-install
modify_date: "2021-03-09"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### Getting started

Spark NLP for Healthcare is a commercial extension of Spark NLP for clinical and biomedical text mining. If you don't have a Spark NLP for Healthcare subscription yet, you can ask for a free trial by clicking on the button below.

{:.btn-block}
[Try Free](https://www.johnsnowlabs.com/spark-nlp-try-free/){:.button.button--primary.button--rounded.button--lg}


Spark NLP for Healthcare provides healthcare-specific annotators, pipelines, models, and embeddings for:
- Clinical entity recognition
- Clinical Entity Linking
- Entity normalization
- Assertion Status Detection
- De-identification
- Relation Extraction
- Spell checking & correction

note: If you are going to use any pretrained licensed NER model, you don't need to install licensed libray. As long as you have the AWS keys and license keys in your environment, you will be able to use licensed NER models with Spark NLP public library. For the other licensed pretrained models like AssertionDL, Deidentification, Entity Resolvers and Relation Extraction models, you will need to install Spark NLP Enterprise as well.

The library offers access to several clinical and biomedical transformers: JSL-BERT-Clinical, BioBERT, ClinicalBERT, GloVe-Med, GloVe-ICD-O. It also includes over 50 pre-trained healthcare models, that can recognize the following entities (any many more):
- Clinical - support Signs, Symptoms, Treatments, Procedures, Tests, Labs, Sections
- Drugs - support Name, Dosage, Strength, Route, Duration, Frequency
- Risk Factors- support Smoking, Obesity, Diabetes, Hypertension, Substance Abuse
- Anatomy - support Organ, Subdivision, Cell, Structure Organism, Tissue, Gene, Chemical
- Demographics - support Age, Gender, Height, Weight, Race, Ethnicity, Marital Status, Vital Signs
- Sensitive Data- support Patient Name, Address, Phone, Email, Dates, Providers, Identifiers
