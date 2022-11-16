---
layout: model
title: Pipeline to Resolve ICD-9-CM Codes
author: John Snow Labs
name: icd9_resolver_pipeline
date: 2022-09-30
tags: [en, licensed, clinical, resolver, chunk_mapping, pipeline, icd9cm]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities with their corresponding ICD-9-CM codes. You’ll just feed your text and it will return the corresponding ICD-9-CM codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd9_resolver_pipeline_en_4.1.0_3.0_1664543263329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

resolver_pipeline = PretrainedPipeline("icd9_resolver_pipeline", "en", "clinical/models")

text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage"""

result = resolver_pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val med_resolver_pipeline = new PretrainedPipeline("icd9_resolver_pipeline", "en", "clinical/models")

val result = med_resolver_pipeline.fullAnnotate("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage""")
```
</div>

## Results

```bash
+-----------------------------+---------+---------+
|chunk                        |ner_chunk|icd9_code|
+-----------------------------+---------+---------+
|gestational diabetes mellitus|PROBLEM  |V12.21   |
|anisakiasis                  |PROBLEM  |127.1    |
|fetal and neonatal hemorrhage|PROBLEM  |772      |
+-----------------------------+---------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd9_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|2.2 GB|

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