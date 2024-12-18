---
layout: model
title: English mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline pipeline DeBertaForSequenceClassification from quancute
author: John Snow Labs
name: mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline
date: 2024-12-17
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline` is a English model originally trained by quancute.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline_en_5.5.1_3.0_1734405523766.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline_en_5.5.1_3.0_1734405523766.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_ner_ghtk_hirach_ner_first_1000_data_3090_15nov_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|814.8 MB|

## References

https://huggingface.co/quancute/mdeberta-ner-ghtk-hirach_NER-first_1000_data-3090-15Nov

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification