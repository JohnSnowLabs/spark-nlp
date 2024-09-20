---
layout: model
title: Multilingual multilingual_xlm_roberta_for_ner_eserdy_pipeline pipeline XlmRoBertaForTokenClassification from eserdy
author: John Snow Labs
name: multilingual_xlm_roberta_for_ner_eserdy_pipeline
date: 2024-09-07
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_xlm_roberta_for_ner_eserdy_pipeline` is a Multilingual model originally trained by eserdy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_xlm_roberta_for_ner_eserdy_pipeline_xx_5.5.0_3.0_1725694256321.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_xlm_roberta_for_ner_eserdy_pipeline_xx_5.5.0_3.0_1725694256321.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilingual_xlm_roberta_for_ner_eserdy_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilingual_xlm_roberta_for_ner_eserdy_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_xlm_roberta_for_ner_eserdy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|840.8 MB|

## References

https://huggingface.co/eserdy/multilingual-xlm-roberta-for-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification