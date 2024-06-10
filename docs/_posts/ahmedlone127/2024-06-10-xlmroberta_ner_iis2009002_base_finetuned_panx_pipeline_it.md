---
layout: model
title: Italian xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline pipeline XlmRoBertaForTokenClassification from iis2009002
author: John Snow Labs
name: xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline
date: 2024-06-10
tags: [it, open_source, pipeline, onnx]
task: Named Entity Recognition
language: it
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline` is a Italian model originally trained by iis2009002.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline_it_5.4.0_3.0_1718024760262.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline_it_5.4.0_3.0_1718024760262.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_iis2009002_base_finetuned_panx_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|828.6 MB|

## References

https://huggingface.co/iis2009002/xlm-roberta-base-finetuned-panx-it

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification