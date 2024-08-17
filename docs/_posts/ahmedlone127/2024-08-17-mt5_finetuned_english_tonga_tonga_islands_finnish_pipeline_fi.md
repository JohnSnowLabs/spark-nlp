---
layout: model
title: Finnish mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline pipeline T5Transformer from ElliottZ
author: John Snow Labs
name: mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline
date: 2024-08-17
tags: [fi, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline` is a Finnish model originally trained by ElliottZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline_fi_5.4.2_3.0_1723906137127.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline_fi_5.4.2_3.0_1723906137127.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_finetuned_english_tonga_tonga_islands_finnish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|819.8 MB|

## References

https://huggingface.co/ElliottZ/mt5-finetuned-english-to-Finnish

## Included Models

- DocumentAssembler
- T5Transformer