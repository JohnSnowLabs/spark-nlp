---
layout: model
title: Pipeline to Resolve ICD-10-CM Codes
author: John Snow Labs
name: icd10cm_resolver_pipeline
date: 2022-11-02
tags: [en, clinical, licensed, resolver, chunk_mapping, pipeline, icd10cm]
task: Pipeline Healthcare
language: en
edition: Spark NLP for Healthcare 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities with their corresponding ICD-10-CM codes. You’ll just feed your text and it will return the corresponding ICD-10-CM codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_resolver_pipeline_en_4.2.1_3.0_1667389014041.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10cm_resolver_pipeline_en_4.2.1_3.0_1667389014041.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

resolver_pipeline = PretrainedPipeline("icd10cm_resolver_pipeline", "en", "clinical/models")

text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage"""

result = resolver_pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val resolver_pipeline = new PretrainedPipeline("icd10cm_resolver_pipeline", "en", "clinical/models")

val result = resolver_pipeline.fullAnnotate("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage""")
```
</div>

## Results

```bash
+-----------------------------+---------+------------+
|chunk                        |ner_chunk|icd10cm_code|
+-----------------------------+---------+------------+
|gestational diabetes mellitus|PROBLEM  |O24.919     |
|anisakiasis                  |PROBLEM  |B81.0       |
|fetal and neonatal hemorrhage|PROBLEM  |P545        |
+-----------------------------+---------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|3.5 GB|

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