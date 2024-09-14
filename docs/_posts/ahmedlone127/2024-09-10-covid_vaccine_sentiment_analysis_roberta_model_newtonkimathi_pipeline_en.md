---
layout: model
title: English covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline pipeline RoBertaForSequenceClassification from NewtonKimathi
author: John Snow Labs
name: covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline
date: 2024-09-10
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline` is a English model originally trained by NewtonKimathi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline_en_5.5.0_3.0_1725962379341.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline_en_5.5.0_3.0_1725962379341.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|covid_vaccine_sentiment_analysis_roberta_model_newtonkimathi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|436.0 MB|

## References

https://huggingface.co/NewtonKimathi/Covid_Vaccine_Sentiment_Analysis_Roberta_Model

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification