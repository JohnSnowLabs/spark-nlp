---
layout: model
title: Sentence Detection in Indonesian Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [open_source, sentence_detection, id]
task: Sentence Detection
language: id
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_id_3.2.0_3.0_1630318954338.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_id_3.2.0_3.0_1630318954338.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "id") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Mencari sumber paragraf bacaan bahasa Inggris yang bagus? Anda telah datang ke tempat yang tepat. Menurut sebuah penelitian baru-baru ini, kebiasaan membaca di kalangan remaja saat ini menurun dengan cepat. Mereka tidak dapat fokus pada paragraf bacaan bahasa Inggris yang diberikan selama lebih dari beberapa detik! Juga, membaca adalah dan merupakan bagian integral dari semua ujian kompetitif. Jadi, bagaimana Anda meningkatkan keterampilan membaca Anda? Jawaban atas pertanyaan ini sebenarnya adalah pertanyaan lain: Apa gunanya keterampilan membaca? Tujuan utama membaca adalah 'untuk masuk akal'.""")


```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "id")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("Mencari sumber paragraf bacaan bahasa Inggris yang bagus? Anda telah datang ke tempat yang tepat. Menurut sebuah penelitian baru-baru ini, kebiasaan membaca di kalangan remaja saat ini menurun dengan cepat. Mereka tidak dapat fokus pada paragraf bacaan bahasa Inggris yang diberikan selama lebih dari beberapa detik! Juga, membaca adalah dan merupakan bagian integral dari semua ujian kompetitif. Jadi, bagaimana Anda meningkatkan keterampilan membaca Anda? Jawaban atas pertanyaan ini sebenarnya adalah pertanyaan lain: Apa gunanya keterampilan membaca? Tujuan utama membaca adalah 'untuk masuk akal'.").toDF("text")
val result = pipeline.fit(data).transform(data)


```

{:.nlu-block}
```python
nlu.load('id.sentence_detector').predict("Mencari sumber paragraf bacaan bahasa Inggris yang bagus? Anda telah datang ke tempat yang tepat. Menurut sebuah penelitian baru-baru ini, kebiasaan membaca di kalangan remaja saat ini menurun dengan cepat. Mereka tidak dapat fokus pada paragraf bacaan bahasa Inggris yang diberikan selama lebih dari beberapa detik! Juga, membaca adalah dan merupakan bagian integral dari semua ujian kompetitif. Jadi, bagaimana Anda meningkatkan keterampilan membaca Anda? Jawaban atas pertanyaan ini sebenarnya adalah pertanyaan lain: Apa gunanya keterampilan membaca? Tujuan utama membaca adalah 'untuk masuk akal'.", output_level ='sentence')  
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------------+
|result                                                                                                         |
+---------------------------------------------------------------------------------------------------------------+
|[Mencari sumber paragraf bacaan bahasa Inggris yang bagus?]                                                    |
|[Anda telah datang ke tempat yang tepat.]                                                                      |
|[Menurut sebuah penelitian baru-baru ini, kebiasaan membaca di kalangan remaja saat ini menurun dengan cepat.] |
|[Mereka tidak dapat fokus pada paragraf bacaan bahasa Inggris yang diberikan selama lebih dari beberapa detik!]|
|[Juga, membaca adalah dan merupakan bagian integral dari semua ujian kompetitif.]                              |
|[Jadi, bagaimana Anda meningkatkan keterampilan membaca Anda?]                                                 |
|[Jawaban atas pertanyaan ini sebenarnya adalah pertanyaan lain:]                                               |
|[Apa gunanya keterampilan membaca?]                                                                            |
|[Tujuan utama membaca adalah 'untuk masuk akal'.]                                                              |
+---------------------------------------------------------------------------------------------------------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|id|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
