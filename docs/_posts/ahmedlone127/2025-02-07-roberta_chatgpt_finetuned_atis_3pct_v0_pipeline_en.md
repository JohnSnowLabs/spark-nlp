---
layout: model
title: English roberta_chatgpt_finetuned_atis_3pct_v0_pipeline pipeline RoBertaForSequenceClassification from benayas
author: John Snow Labs
name: roberta_chatgpt_finetuned_atis_3pct_v0_pipeline
date: 2025-02-07
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_chatgpt_finetuned_atis_3pct_v0_pipeline` is a English model originally trained by benayas.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_chatgpt_finetuned_atis_3pct_v0_pipeline_en_5.5.1_3.0_1738920054221.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_chatgpt_finetuned_atis_3pct_v0_pipeline_en_5.5.1_3.0_1738920054221.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_chatgpt_finetuned_atis_3pct_v0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_chatgpt_finetuned_atis_3pct_v0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_chatgpt_finetuned_atis_3pct_v0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|426.8 MB|

## References

https://huggingface.co/benayas/roberta-chatgpt-finetuned-atis_3pct_v0

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification