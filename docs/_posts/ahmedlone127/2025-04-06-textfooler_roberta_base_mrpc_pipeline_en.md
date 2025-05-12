---
layout: model
title: English textfooler_roberta_base_mrpc_pipeline pipeline RoBertaForSequenceClassification from korca
author: John Snow Labs
name: textfooler_roberta_base_mrpc_pipeline
date: 2025-04-06
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`textfooler_roberta_base_mrpc_pipeline` is a English model originally trained by korca.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/textfooler_roberta_base_mrpc_pipeline_en_5.5.1_3.0_1743921212476.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/textfooler_roberta_base_mrpc_pipeline_en_5.5.1_3.0_1743921212476.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("textfooler_roberta_base_mrpc_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("textfooler_roberta_base_mrpc_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|textfooler_roberta_base_mrpc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|450.6 MB|

## References

https://huggingface.co/korca/textfooler-roberta-base-mrpc

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification