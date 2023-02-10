---
layout: model
title: Pipeline to Resolve CVX Codes
author: John Snow Labs
name: cvx_resolver_pipeline
date: 2022-10-12
tags: [en, licensed, clinical, resolver, chunk_mapping, cvx, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities with their corresponding CVX codes. You’ll just feed your text and it will return the corresponding CVX codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/cvx_resolver_pipeline_en_4.2.1_3.0_1665611325640.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/cvx_resolver_pipeline_en_4.2.1_3.0_1665611325640.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

resolver_pipeline = PretrainedPipeline("cvx_resolver_pipeline", "en", "clinical/models")

text= "The patient has a history of influenza vaccine, tetanus and DTaP"

result = resolver_pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val resolver_pipeline = new PretrainedPipeline("cvx_resolver_pipeline", "en", "clinical/models")

val result = resolver_pipeline.fullAnnotate("The patient has a history of influenza vaccine, tetanus and DTaP")
```
</div>

## Results

```bash
+-----------------+---------+--------+
|chunk            |ner_chunk|cvx_code|
+-----------------+---------+--------+
|influenza vaccine|Vaccine  |160     |
|tetanus          |Vaccine  |35      |
|DTaP             |Vaccine  |20      |
+-----------------+---------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cvx_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|2.1 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger
