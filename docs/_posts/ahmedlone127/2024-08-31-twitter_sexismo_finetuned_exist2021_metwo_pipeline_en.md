---
layout: model
title: English twitter_sexismo_finetuned_exist2021_metwo_pipeline pipeline RoBertaForSequenceClassification from somosnlp-hackathon-2022
author: John Snow Labs
name: twitter_sexismo_finetuned_exist2021_metwo_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`twitter_sexismo_finetuned_exist2021_metwo_pipeline` is a English model originally trained by somosnlp-hackathon-2022.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/twitter_sexismo_finetuned_exist2021_metwo_pipeline_en_5.4.2_3.0_1725122615360.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/twitter_sexismo_finetuned_exist2021_metwo_pipeline_en_5.4.2_3.0_1725122615360.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("twitter_sexismo_finetuned_exist2021_metwo_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("twitter_sexismo_finetuned_exist2021_metwo_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_sexismo_finetuned_exist2021_metwo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.3 MB|

## References

https://huggingface.co/somosnlp-hackathon-2022/twitter_sexismo-finetuned-exist2021-metwo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification