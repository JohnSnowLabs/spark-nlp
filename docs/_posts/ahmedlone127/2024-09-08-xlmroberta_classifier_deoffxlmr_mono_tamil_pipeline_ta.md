---
layout: model
title: Tamil xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline pipeline XlmRoBertaForSequenceClassification from Hate-speech-CNERG
author: John Snow Labs
name: xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline
date: 2024-09-08
tags: [ta, open_source, pipeline, onnx]
task: Text Classification
language: ta
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline` is a Tamil model originally trained by Hate-speech-CNERG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline_ta_5.5.0_3.0_1725781627635.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline_ta_5.5.0_3.0_1725781627635.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_classifier_deoffxlmr_mono_tamil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|1.0 GB|

## References

https://huggingface.co/Hate-speech-CNERG/deoffxlmr-mono-tamil

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification