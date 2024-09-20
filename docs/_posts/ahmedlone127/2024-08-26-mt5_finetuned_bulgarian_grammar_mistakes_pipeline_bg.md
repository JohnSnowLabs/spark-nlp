---
layout: model
title: Bulgarian mt5_finetuned_bulgarian_grammar_mistakes_pipeline pipeline T5Transformer from thebogko
author: John Snow Labs
name: mt5_finetuned_bulgarian_grammar_mistakes_pipeline
date: 2024-08-26
tags: [bg, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: bg
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_finetuned_bulgarian_grammar_mistakes_pipeline` is a Bulgarian model originally trained by thebogko.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_finetuned_bulgarian_grammar_mistakes_pipeline_bg_5.4.2_3.0_1724711008021.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_finetuned_bulgarian_grammar_mistakes_pipeline_bg_5.4.2_3.0_1724711008021.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_finetuned_bulgarian_grammar_mistakes_pipeline", lang = "bg")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_finetuned_bulgarian_grammar_mistakes_pipeline", lang = "bg")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_finetuned_bulgarian_grammar_mistakes_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|bg|
|Size:|2.3 GB|

## References

https://huggingface.co/thebogko/mt5-finetuned-bulgarian-grammar-mistakes

## Included Models

- DocumentAssembler
- T5Transformer