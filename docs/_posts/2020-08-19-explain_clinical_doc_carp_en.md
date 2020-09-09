---
layout: model
title: Explain Clinical Doc CARP
author: John Snow Labs
name: explain_clinical_doc_carp_en
date: 2020-08-19
tags: [pipeline, en, explain_clinical_doc_carp]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A pretrained pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.


{:.btn-box}
[Live Demo](){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_carp_en_2.5.5_2.4_1597841630062.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

{% include programmingLanguageSelectScalaPython.html %}

```python


pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

annotations = pipeline.annotate(text)

annotations.keys()

```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_carp_en_2.5.5_2.4|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.5.5|
|Edition:|Healthcare|
|License:|Enterprise|
|Language:|[en]|

{:.h2_title}
## Included Models 
 - ner_clinical
 - assertion_dl
 - re_clinical
 - ner_posology
{:.h2_title}
## Results

The output is a dictionary with the following keys: 'sentences', 'clinical_ner_tags', 'document', 'ner_chunks', 'clinical_ner_chunks', 'ner_tags', 'assertion', 'clinical_relations', 'tokens', 'embeddings', 'pos_tags', 'dependencies'.
