---
layout: model
title: Diseases and Syndromes to UMLS Code Mapping
author: John Snow Labs
name: umls_disease_syndrome_mapping
date: 2022-07-25
tags: [umls, en, licensed, pipeline]
task: Pipeline Healthcare
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities (Diseases and Syndromes) with their corresponding UMLS CUI codes. You’ll just feed your text and it will return the corresponding UMLS codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/umls_disease_syndrome_mapping_en_4.0.0_3.0_1658730673896.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("umls_disease_syndrome_mapping", "en", "clinical/models")
pipeline.annotate("A 34-year-old female with a history of poor appetite, gestational diabetes mellitus, acyclovir allergy and polyuria")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= PretrainedPipeline("umls_disease_syndrome_mapping", "en", "clinical/models")
val pipeline.annotate("A 34-year-old female with a history of poor appetite, gestational diabetes mellitus, acyclovir allergy and polyuria")
```
</div>

## Results

```bash
+---------------------------------------------------------------------------+----------------------------------------+
|chunks                                                                     |umls_code                               |
+---------------------------------------------------------------------------+----------------------------------------+
|[poor appetite, gestational diabetes mellitus, acyclovir allergy, polyuria]|[C0003123, C0085207, C0571297, C0018965]|
+---------------------------------------------------------------------------+----------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umls_disease_syndrome_mapping|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|3.4 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger