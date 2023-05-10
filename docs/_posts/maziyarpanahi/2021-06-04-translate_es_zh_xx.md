---
layout: model
title: Translate Spanish to Chinese Pipeline
author: John Snow Labs
name: translate_es_zh
date: 2021-06-04
tags: [open_source, pipeline, seq2seq, translation, es, zh, xx, multilingual]
task: Translation
language: xx
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.
It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

source languages: es

target languages: zh

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/TRANSLATION_MARIAN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_MARIAN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/translate_es_zh_xx_3.1.0_2.4_1622843159186.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/translate_es_zh_xx_3.1.0_2.4_1622843159186.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("translate_es_zh", lang = "xx") 
pipeline.annotate("Your sentence to translate!")
```
```scala

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("translate_es_zh", lang = "xx")
pipeline.annotate("Your sentence to translate!")
```

{:.nlu-block}
```python

import nlu
text = ["text to translate"]
translate_df = nlu.load('xx.Spanish.translate_to.Chinese').predict(text, output_level='sentence')
translate_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|translate_es_zh|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|

## Data Source

[https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models)

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer