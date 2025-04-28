---
layout: model
title: English topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline pipeline RoBertaForSequenceClassification from tweettemposhift
author: John Snow Labs
name: topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline
date: 2025-04-02
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline` is a English model originally trained by tweettemposhift.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline_en_5.5.1_3.0_1743616515599.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline_en_5.5.1_3.0_1743616515599.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|topic_topic_random3_seed2_twitter_roberta_large_2022_154m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/tweettemposhift/topic-topic_random3_seed2-twitter-roberta-large-2022-154m

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification