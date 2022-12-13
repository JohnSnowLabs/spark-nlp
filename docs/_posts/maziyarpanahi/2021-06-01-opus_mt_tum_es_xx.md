---
layout: model
title: Fast Neural Machine Translation Model from Tumbuka to Spanish
author: John Snow Labs
name: opus_mt_tum_es
date: 2021-06-01
tags: [open_source, seq2seq, translation, tum, es, xx, multilingual]
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

source languages: tum

target languages: es

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_mt_tum_es_xx_3.1.0_2.4_1622551468089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_mt_tum_es_xx_3.1.0_2.4_1622551468089.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\ 
.setInputCol("text")\ 
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
.setInputCols(["document"])\ 
.setOutputCol("sentences")

marian = MarianTransformer.pretrained("opus_mt_tum_es", "xx")\ 
.setInputCols(["sentence"])\ 
.setOutputCol("translation")
```
```scala

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val marian = MarianTransformer.pretrained("opus_mt_tum_es", "xx")
.setInputCols(["sentence"])
.setOutputCol("translation")
```

{:.nlu-block}
```python

import nlu
text = ["text to translate"]
translate_df = nlu.load('xx.Tumbuka.translate_to.Spanish').predict(text, output_level='sentence')
translate_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_mt_tum_es|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[translation]|
|Language:|xx|

## Data Source

[https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models)