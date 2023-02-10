---
layout: model
title: Fast Neural Machine Translation Model from English to Umbundu
author: John Snow Labs
name: opus_mt_en_umb
date: 2020-12-28
task: Translation
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, seq2seq, translation, en, umb, xx]
supported: true
annotator: MarianTransformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.
It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

- source languages: `en`

- target languages: `umb`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_mt_en_umb_xx_2.7.0_2.4_1609163118124.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_mt_en_umb_xx_2.7.0_2.4_1609163118124.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

marian = MarianTransformer.pretrained("opus_mt_en_umb", "xx")\ 
.setInputCols(["sentence"])\ 
.setOutputCol("translation")

marian_pipeline = Pipeline(stages=[documentAssembler, sentencerDL, marian])
light_pipeline = LightPipeline(marian_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))
result = light_pipeline.fullAnnotate(data)
```
```scala

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val marian = MarianTransformer.pretrained("opus_mt_en_umb", "xx")
.setInputCols(["sentence"])
.setOutputCol("translation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, marian))
val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["text to translate"]
opus_df = nlu.load('xx.en.marian.translate_to.umb').predict(text, output_level='sentence')
opus_df
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_mt_en_umb|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[translation]|
|Language:|xx|

## Data Source

[https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models)