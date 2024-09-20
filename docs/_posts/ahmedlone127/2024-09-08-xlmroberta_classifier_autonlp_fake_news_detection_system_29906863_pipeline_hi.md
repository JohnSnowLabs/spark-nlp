---
layout: model
title: Hindi xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline pipeline XlmRoBertaForSequenceClassification from rohansingh
author: John Snow Labs
name: xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline
date: 2024-09-08
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline` is a Hindi model originally trained by rohansingh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline_hi_5.5.0_3.0_1725780850195.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline_hi_5.5.0_3.0_1725780850195.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_classifier_autonlp_fake_news_detection_system_29906863_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|786.9 MB|

## References

https://huggingface.co/rohansingh/autonlp-Fake-news-detection-system-29906863

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification