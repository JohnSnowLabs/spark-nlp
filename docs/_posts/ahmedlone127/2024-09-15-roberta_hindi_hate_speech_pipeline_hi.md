---
layout: model
title: Hindi roberta_hindi_hate_speech_pipeline pipeline RoBertaForSequenceClassification from arnabmukhopadhyay
author: John Snow Labs
name: roberta_hindi_hate_speech_pipeline
date: 2024-09-15
tags: [hi, open_source, pipeline, onnx]
task: Text Classification
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_hindi_hate_speech_pipeline` is a Hindi model originally trained by arnabmukhopadhyay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_hindi_hate_speech_pipeline_hi_5.5.0_3.0_1726401902089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_hindi_hate_speech_pipeline_hi_5.5.0_3.0_1726401902089.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_hindi_hate_speech_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_hindi_hate_speech_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_hindi_hate_speech_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|467.2 MB|

## References

https://huggingface.co/arnabmukhopadhyay/Roberta-hindi-hate-speech

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification