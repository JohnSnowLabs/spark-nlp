---
layout: model
title: Multilingual reward_deberta_v3_large_aspect_pipeline pipeline DeBertaForSequenceClassification from theblackcat102
author: John Snow Labs
name: reward_deberta_v3_large_aspect_pipeline
date: 2024-09-02
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`reward_deberta_v3_large_aspect_pipeline` is a Multilingual model originally trained by theblackcat102.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/reward_deberta_v3_large_aspect_pipeline_xx_5.5.0_3.0_1725282119939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/reward_deberta_v3_large_aspect_pipeline_xx_5.5.0_3.0_1725282119939.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("reward_deberta_v3_large_aspect_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("reward_deberta_v3_large_aspect_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|reward_deberta_v3_large_aspect_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.0 GB|

## References

https://huggingface.co/theblackcat102/reward-deberta-v3-large-aspect

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification