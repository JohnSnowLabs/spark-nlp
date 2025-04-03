---
layout: model
title: Multilingual egd_distilbert_base_multilingual_cased_pipeline pipeline DistilBertForSequenceClassification from uvegesistvan
author: John Snow Labs
name: egd_distilbert_base_multilingual_cased_pipeline
date: 2025-04-02
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`egd_distilbert_base_multilingual_cased_pipeline` is a Multilingual model originally trained by uvegesistvan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/egd_distilbert_base_multilingual_cased_pipeline_xx_5.5.1_3.0_1743563386316.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/egd_distilbert_base_multilingual_cased_pipeline_xx_5.5.1_3.0_1743563386316.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("egd_distilbert_base_multilingual_cased_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("egd_distilbert_base_multilingual_cased_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|egd_distilbert_base_multilingual_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|507.7 MB|

## References

https://huggingface.co/uvegesistvan/EGD_distilbert-base-multilingual-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification