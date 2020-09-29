---
layout: model
title: Explain Clinical Doc CARP
author: John Snow Labs
name: explain_clinical_doc_carp
date: 2020-08-19
tags: [pipeline, en, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A pretrained pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_carp_en_2.5.5_2.4_1597841630062.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

annotations = pipeline.annotate("This is an example")

```

{:.noactive}
```scala
```
</div>

{:.h2_title}
## Results
```bash
{'sentences': ['This is an example'],
 'clinical_ner_tags': ['O', 'O', 'O', 'O'],
 'document': ['This is an example'],
 'ner_chunks': [],
 'clinical_ner_chunks': [],
 'ner_tags': ['O', 'O', 'O', 'O'],
 'assertion': [],
 'clinical_relations': [],
 'tokens': ['This', 'is', 'an', 'example'],
 'embeddings': ['This', 'is', 'an', 'example'],
 'pos_tags': ['PND', 'VBZ', 'DD', 'NN'],
 'dependencies': ['example', 'example', 'example', 'ROOT']}
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_carp|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.5.5|
|License:|Licensed|
|Edition:|Official|
|Language:|[en]|

{:.h2_title}
## Included Models 
 - ner_clinical
 - assertion_dl
 - re_clinical
 - ner_posology
 
