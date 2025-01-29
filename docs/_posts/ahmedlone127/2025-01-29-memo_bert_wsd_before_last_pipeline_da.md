---
layout: model
title: Danish memo_bert_wsd_before_last_pipeline pipeline XlmRoBertaForSequenceClassification from yemen2016
author: John Snow Labs
name: memo_bert_wsd_before_last_pipeline
date: 2025-01-29
tags: [da, open_source, pipeline, onnx]
task: Text Classification
language: da
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`memo_bert_wsd_before_last_pipeline` is a Danish model originally trained by yemen2016.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/memo_bert_wsd_before_last_pipeline_da_5.5.1_3.0_1738125473251.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/memo_bert_wsd_before_last_pipeline_da_5.5.1_3.0_1738125473251.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("memo_bert_wsd_before_last_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("memo_bert_wsd_before_last_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|memo_bert_wsd_before_last_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|466.6 MB|

## References

https://huggingface.co/yemen2016/MeMo-BERT-WSD_before_last

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification