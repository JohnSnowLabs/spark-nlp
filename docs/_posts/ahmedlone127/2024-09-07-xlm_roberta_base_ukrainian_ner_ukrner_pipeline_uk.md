---
layout: model
title: Ukrainian xlm_roberta_base_ukrainian_ner_ukrner_pipeline pipeline XlmRoBertaForTokenClassification from EvanD
author: John Snow Labs
name: xlm_roberta_base_ukrainian_ner_ukrner_pipeline
date: 2024-09-07
tags: [uk, open_source, pipeline, onnx]
task: Named Entity Recognition
language: uk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_ukrainian_ner_ukrner_pipeline` is a Ukrainian model originally trained by EvanD.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_ukrainian_ner_ukrner_pipeline_uk_5.5.0_3.0_1725743325332.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_ukrainian_ner_ukrner_pipeline_uk_5.5.0_3.0_1725743325332.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_ukrainian_ner_ukrner_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_ukrainian_ner_ukrner_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_ukrainian_ner_ukrner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|785.4 MB|

## References

https://huggingface.co/EvanD/xlm-roberta-base-ukrainian-ner-ukrner

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification