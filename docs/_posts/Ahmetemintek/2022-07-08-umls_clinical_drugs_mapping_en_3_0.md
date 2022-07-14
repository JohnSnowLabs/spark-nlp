---
layout: model
title: Clinical Drugs to UMLS Code Mapping
author: John Snow Labs
name: umls_clinical_drugs_mapping
date: 2022-07-08
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

This pretrained pipeline maps entities (Clinical Drugs) with their corresponding UMLS CUI codes. You’ll just feed your text and it will return the corresponding UMLS codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/umls_clinical_drugs_mapping_en_4.0.0_3.0_1657289917698.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("umls_clinical_drugs_mapping", "en", "clinical/models")
pipeline.annotate("The patient was given Adapin 10 MG, coumadn 5 mg")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("umls_clinical_drugs_mapping","en","clinical/models")
val result = pipeline.annotate("The patient was given Adapin 10 MG, coumadn 5 mg")
```
</div>

## Results

```bash
+-----------------+--------------------+
|chunks           |umls_code           |
+-----------------+--------------------+
|[Adapin, coumadn]|[C0728756, C0591275]|
+-----------------+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umls_clinical_drugs_mapping|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|4.6 GB|

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