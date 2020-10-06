---
layout: model
title: Extract clinical entities, assign assertion, and find relations.
author: John Snow Labs
name: explain_clinical_doc_era
date: 2020-09-30
tags: [pipeline, en, licensed]
article_header:
  type: cover
use_language_switcher: "Python"
---

## Description
A pretrained pipeline with ner_clinical_events, assertion_dl and re_temporal_events_clinical trained with embeddings_healthcare_100d. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_2.5.5_2.4_1597845753750.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('explain_clinical_doc_era', 'en', 'clinical/models')

annotations = pipeline.annotate(text)

annotations.keys()

```

</div>

{:.h2_title}
## Results
The output is a dictionary with the following keys: 'sentences', 'clinical_ner_tags', 'clinical_ner_chunks_re', 'document', 'clinical_ner_chunks', 'assertion', 'clinical_relations', 'tokens', 'embeddings', 'pos_tags', 'dependencies'.

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_era|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|License:|Licensed|
|Edition:|Official|
|Language:|[en]|

{:.h2_title}
## Included Models 
 - ner_clinical_events
 - assertion_dl
 - re_temporal_events_clinical
 
