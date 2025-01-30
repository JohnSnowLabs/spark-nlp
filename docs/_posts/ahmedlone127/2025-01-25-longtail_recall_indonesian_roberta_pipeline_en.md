---
layout: model
title: English longtail_recall_indonesian_roberta_pipeline pipeline XlmRoBertaForSequenceClassification from yzhang0112
author: John Snow Labs
name: longtail_recall_indonesian_roberta_pipeline
date: 2025-01-25
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`longtail_recall_indonesian_roberta_pipeline` is a English model originally trained by yzhang0112.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longtail_recall_indonesian_roberta_pipeline_en_5.5.1_3.0_1737816682850.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/longtail_recall_indonesian_roberta_pipeline_en_5.5.1_3.0_1737816682850.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("longtail_recall_indonesian_roberta_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("longtail_recall_indonesian_roberta_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longtail_recall_indonesian_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|827.9 MB|

## References

https://huggingface.co/yzhang0112/longtail_recall_id_roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification