---
layout: model
title: English finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline pipeline RoBertaForSequenceClassification from Mariamtc
author: John Snow Labs
name: finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline
date: 2025-02-08
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline` is a English model originally trained by Mariamtc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline_en_5.5.1_3.0_1738986709741.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline_en_5.5.1_3.0_1738986709741.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_twitter_roberta_base_sep2022_tweetcognition_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.4 MB|

## References

https://huggingface.co/Mariamtc/finetuned-twitter-roberta-base-sep2022-tweetcognition

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification