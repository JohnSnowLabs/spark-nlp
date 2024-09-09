---
layout: model
title: English ner_ner_random2_seed1_bernice_pipeline pipeline XlmRoBertaForTokenClassification from tweettemposhift
author: John Snow Labs
name: ner_ner_random2_seed1_bernice_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_ner_random2_seed1_bernice_pipeline` is a English model originally trained by tweettemposhift.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ner_random2_seed1_bernice_pipeline_en_5.5.0_3.0_1725895371899.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_ner_random2_seed1_bernice_pipeline_en_5.5.0_3.0_1725895371899.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ner_ner_random2_seed1_bernice_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ner_ner_random2_seed1_bernice_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ner_random2_seed1_bernice_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|802.6 MB|

## References

https://huggingface.co/tweettemposhift/ner-ner_random2_seed1-bernice

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification