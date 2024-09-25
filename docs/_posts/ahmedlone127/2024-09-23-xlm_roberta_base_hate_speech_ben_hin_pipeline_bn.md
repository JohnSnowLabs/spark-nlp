---
layout: model
title: Bengali xlm_roberta_base_hate_speech_ben_hin_pipeline pipeline XlmRoBertaForSequenceClassification from kingshukroy
author: John Snow Labs
name: xlm_roberta_base_hate_speech_ben_hin_pipeline
date: 2024-09-23
tags: [bn, open_source, pipeline, onnx]
task: Text Classification
language: bn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_hate_speech_ben_hin_pipeline` is a Bengali model originally trained by kingshukroy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_hate_speech_ben_hin_pipeline_bn_5.5.0_3.0_1727089444477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_hate_speech_ben_hin_pipeline_bn_5.5.0_3.0_1727089444477.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_hate_speech_ben_hin_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_hate_speech_ben_hin_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_hate_speech_ben_hin_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|791.2 MB|

## References

https://huggingface.co/kingshukroy/xlm-roberta-base-hate-speech-ben-hin

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification