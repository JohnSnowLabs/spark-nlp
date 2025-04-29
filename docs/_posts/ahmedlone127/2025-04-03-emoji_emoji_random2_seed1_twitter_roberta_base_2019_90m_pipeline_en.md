---
layout: model
title: English emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline pipeline RoBertaForSequenceClassification from tweettemposhift
author: John Snow Labs
name: emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline
date: 2025-04-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline` is a English model originally trained by tweettemposhift.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline_en_5.5.1_3.0_1743653992431.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline_en_5.5.1_3.0_1743653992431.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|emoji_emoji_random2_seed1_twitter_roberta_base_2019_90m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.6 MB|

## References

https://huggingface.co/tweettemposhift/emoji-emoji_random2_seed1-twitter-roberta-base-2019-90m

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification