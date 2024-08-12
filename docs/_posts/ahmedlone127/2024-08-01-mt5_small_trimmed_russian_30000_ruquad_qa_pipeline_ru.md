---
layout: model
title: Russian mt5_small_trimmed_russian_30000_ruquad_qa_pipeline pipeline T5Transformer from vocabtrimmer
author: John Snow Labs
name: mt5_small_trimmed_russian_30000_ruquad_qa_pipeline
date: 2024-08-01
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_small_trimmed_russian_30000_ruquad_qa_pipeline` is a Russian model originally trained by vocabtrimmer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_small_trimmed_russian_30000_ruquad_qa_pipeline_ru_5.4.2_3.0_1722523729873.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_small_trimmed_russian_30000_ruquad_qa_pipeline_ru_5.4.2_3.0_1722523729873.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_small_trimmed_russian_30000_ruquad_qa_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_small_trimmed_russian_30000_ruquad_qa_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_small_trimmed_russian_30000_ruquad_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|331.9 MB|

## References

https://huggingface.co/vocabtrimmer/mt5-small-trimmed-ru-30000-ruquad-qa

## Included Models

- DocumentAssembler
- T5Transformer