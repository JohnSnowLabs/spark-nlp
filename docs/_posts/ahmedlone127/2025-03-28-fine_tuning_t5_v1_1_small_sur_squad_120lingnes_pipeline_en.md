---
layout: model
title: English fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline pipeline T5Transformer from OussamaNaya
author: John Snow Labs
name: fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline` is a English model originally trained by OussamaNaya.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline_en_5.5.1_3.0_1743183617899.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline_en_5.5.1_3.0_1743183617899.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tuning_t5_v1_1_small_sur_squad_120lingnes_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|292.3 MB|

## References

https://huggingface.co/OussamaNaya/Fine_tuning_t5-v1_1-small_sur_SQuAD_120Lingnes

## Included Models

- DocumentAssembler
- T5Transformer