---
layout: model
title: Clinical Major Concepts to UMLS Code Pipeline
author: John Snow Labs
name: umls_major_concepts_resolver_pipeline
date: 2022-07-25
tags: [en, umls, licensed, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities (Clinical Major Concepts) with their corresponding UMLS CUI codes. You’ll just feed your text and it will return the corresponding UMLS codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/umls_major_concepts_resolver_pipeline_en_4.0.0_3.0_1658736979238.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/umls_major_concepts_resolver_pipeline_en_4.0.0_3.0_1658736979238.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("umls_major_concepts_resolver_pipeline", "en", "clinical/models")
pipeline.annotate("The patient complains of pustules after falling from stairs. She has been advised Arthroscopy by her primary care pyhsician")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= PretrainedPipeline("umls_major_concepts_resolver_pipeline", "en", "clinical/models")
val pipeline.annotate("The patient complains of pustules after falling from stairs. She has been advised Arthroscopy by her primary care pyhsician")
```
</div>

## Results

```bash
+-----------+-----------------------------------+---------+
|chunk      |ner_label                          |umls_code|
+-----------+-----------------------------------+---------+
|pustules   |Sign_or_Symptom                    |C0241157 |
|stairs     |Daily_or_Recreational_Activity     |C4300351 |
|Arthroscopy|Therapeutic_or_Preventive_Procedure|C0179144 |
+-----------+-----------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umls_major_concepts_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|3.0 GB|

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
