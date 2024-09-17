---
layout: model
title: English bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline pipeline RoBertaForSequenceClassification from JerryYanJiang
author: John Snow Labs
name: bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline
date: 2024-09-16
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline` is a English model originally trained by JerryYanJiang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline_en_5.5.0_3.0_1726470327542.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline_en_5.5.0_3.0_1726470327542.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertweet_large_epoch6_batch4_lr2e_05_w0_005_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/JerryYanJiang/bertweet-large_epoch6_batch4_lr2e-05_w0.005

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification