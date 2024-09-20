---
layout: model
title: Turkish bert_base_turkish_cased_snli_turkish_pipeline pipeline BertForZeroShotClassification from emrecan
author: John Snow Labs
name: bert_base_turkish_cased_snli_turkish_pipeline
date: 2024-09-01
tags: [tr, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: tr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_turkish_cased_snli_turkish_pipeline` is a Turkish model originally trained by emrecan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_turkish_cased_snli_turkish_pipeline_tr_5.4.2_3.0_1725202729484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_turkish_cased_snli_turkish_pipeline_tr_5.4.2_3.0_1725202729484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_turkish_cased_snli_turkish_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_turkish_cased_snli_turkish_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_turkish_cased_snli_turkish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|412.3 MB|

## References

https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForZeroShotClassification