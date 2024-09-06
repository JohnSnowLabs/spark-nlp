---
layout: model
title: Hebrew dictabert_ner_pipeline pipeline BertForTokenClassification from dicta-il
author: John Snow Labs
name: dictabert_ner_pipeline
date: 2024-09-05
tags: [he, open_source, pipeline, onnx]
task: Named Entity Recognition
language: he
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dictabert_ner_pipeline` is a Hebrew model originally trained by dicta-il.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dictabert_ner_pipeline_he_5.5.0_3.0_1725511669303.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dictabert_ner_pipeline_he_5.5.0_3.0_1725511669303.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dictabert_ner_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dictabert_ner_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dictabert_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|668.3 MB|

## References

https://huggingface.co/dicta-il/dictabert-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification