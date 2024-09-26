---
layout: model
title: Turkish turkish_base_bert_uncased_offenseval2020_turkish_pipeline pipeline BertForSequenceClassification from atasoglu
author: John Snow Labs
name: turkish_base_bert_uncased_offenseval2020_turkish_pipeline
date: 2024-09-26
tags: [tr, open_source, pipeline, onnx]
task: Text Classification
language: tr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkish_base_bert_uncased_offenseval2020_turkish_pipeline` is a Turkish model originally trained by atasoglu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_base_bert_uncased_offenseval2020_turkish_pipeline_tr_5.5.0_3.0_1727386284253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_base_bert_uncased_offenseval2020_turkish_pipeline_tr_5.5.0_3.0_1727386284253.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkish_base_bert_uncased_offenseval2020_turkish_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkish_base_bert_uncased_offenseval2020_turkish_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_base_bert_uncased_offenseval2020_turkish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|415.3 MB|

## References

https://huggingface.co/atasoglu/turkish-base-bert-uncased-offenseval2020_tr

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification