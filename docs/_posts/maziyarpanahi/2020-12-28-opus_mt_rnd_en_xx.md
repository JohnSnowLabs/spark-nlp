---
layout: model
title: Fast Neural Machine Translation Model from Ruund to English
author: John Snow Labs
name: opus_mt_rnd_en
date: 2020-12-28
tags: [open_source, seq2seq, translation, rnd, en, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.
            It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

            source languages: rnd

            target languages: en

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_mt_rnd_en_xx_2.7.0_2.4_1609166434564.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
documentAssembler = DocumentAssembler()\ 
 .setInputCol("text")\ 
 .setOutputCol("document")

 sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
 .setInputCols(["document"])\ 
 .setOutputCol("sentences")

 marian = MarianTransformer.pretrained("opus_mt_rnd_en", "xx")\ 
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

                val marian = MarianTransformer.pretrained("opus_mt_rnd_en", "xx")
                .setInputCols(["sentence"])
                .setOutputCol("translation")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_mt_rnd_en|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[translation]|
|Language:|xx|

## Data Source

[https://huggingface.co/Helsinki-NLP/opus-mt-rnd-en](https://huggingface.co/Helsinki-NLP/opus-mt-rnd-en)