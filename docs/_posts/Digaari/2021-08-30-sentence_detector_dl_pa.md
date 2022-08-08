---
layout: model
title: Sentence Detection in Punjabi Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [pa, open_source, sentence_detection]
task: Sentence Detection
language: pa
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_pa_3.2.0_3.0_1630320087911.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "pa") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਦੇ ਪੈਰਾਗ੍ਰਾਫਾਂ ਦੇ ਇੱਕ ਮਹਾਨ ਸਰੋਤ ਦੀ ਭਾਲ ਕਰ ਰਹੇ ਹੋ? ਤੁਸੀਂ ਸਹੀ ਜਗ੍ਹਾ ਤੇ ਆਏ ਹੋ. ਇੱਕ ਤਾਜ਼ਾ ਅਧਿਐਨ ਅਨੁਸਾਰ ਅੱਜ ਦੇ ਨੌਜਵਾਨਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਦੀ ਆਦਤ ਤੇਜ਼ੀ ਨਾਲ ਘਟ ਰਹੀ ਹੈ। ਉਹ ਕੁਝ ਸਕਿੰਟਾਂ ਤੋਂ ਵੱਧ ਸਮੇਂ ਲਈ ਦਿੱਤੇ ਗਏ ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਵਾਲੇ ਪੈਰੇ 'ਤੇ ਧਿਆਨ ਨਹੀਂ ਦੇ ਸਕਦੇ! ਨਾਲ ਹੀ, ਪੜ੍ਹਨਾ ਸਾਰੀਆਂ ਪ੍ਰਤੀਯੋਗੀ ਪ੍ਰੀਖਿਆਵਾਂ ਦਾ ਇੱਕ ਅਨਿੱਖੜਵਾਂ ਅੰਗ ਸੀ ਅਤੇ ਹੈ. ਇਸ ਲਈ, ਤੁਸੀਂ ਆਪਣੇ ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਨੂੰ ਕਿਵੇਂ ਸੁਧਾਰਦੇ ਹੋ? ਇਸ ਪ੍ਰਸ਼ਨ ਦਾ ਉੱਤਰ ਅਸਲ ਵਿੱਚ ਇੱਕ ਹੋਰ ਪ੍ਰਸ਼ਨ ਹੈ: ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਦੀ ਵਰਤੋਂ ਕੀ ਹੈ? ਪੜ੍ਹਨ ਦਾ ਮੁੱਖ ਉਦੇਸ਼ 'ਅਰਥ ਬਣਾਉਣਾ' ਹੈ.""")


```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "pa")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਦੇ ਪੈਰਾਗ੍ਰਾਫਾਂ ਦੇ ਇੱਕ ਮਹਾਨ ਸਰੋਤ ਦੀ ਭਾਲ ਕਰ ਰਹੇ ਹੋ? ਤੁਸੀਂ ਸਹੀ ਜਗ੍ਹਾ ਤੇ ਆਏ ਹੋ. ਇੱਕ ਤਾਜ਼ਾ ਅਧਿਐਨ ਅਨੁਸਾਰ ਅੱਜ ਦੇ ਨੌਜਵਾਨਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਦੀ ਆਦਤ ਤੇਜ਼ੀ ਨਾਲ ਘਟ ਰਹੀ ਹੈ। ਉਹ ਕੁਝ ਸਕਿੰਟਾਂ ਤੋਂ ਵੱਧ ਸਮੇਂ ਲਈ ਦਿੱਤੇ ਗਏ ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਵਾਲੇ ਪੈਰੇ 'ਤੇ ਧਿਆਨ ਨਹੀਂ ਦੇ ਸਕਦੇ! ਨਾਲ ਹੀ, ਪੜ੍ਹਨਾ ਸਾਰੀਆਂ ਪ੍ਰਤੀਯੋਗੀ ਪ੍ਰੀਖਿਆਵਾਂ ਦਾ ਇੱਕ ਅਨਿੱਖੜਵਾਂ ਅੰਗ ਸੀ ਅਤੇ ਹੈ. ਇਸ ਲਈ, ਤੁਸੀਂ ਆਪਣੇ ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਨੂੰ ਕਿਵੇਂ ਸੁਧਾਰਦੇ ਹੋ? ਇਸ ਪ੍ਰਸ਼ਨ ਦਾ ਉੱਤਰ ਅਸਲ ਵਿੱਚ ਇੱਕ ਹੋਰ ਪ੍ਰਸ਼ਨ ਹੈ: ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਦੀ ਵਰਤੋਂ ਕੀ ਹੈ? ਪੜ੍ਹਨ ਦਾ ਮੁੱਖ ਉਦੇਸ਼ 'ਅਰਥ ਬਣਾਉਣਾ' ਹੈ.").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
nlu.load('pa.sentence_detector').predict("ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਦੇ ਪੈਰਾਗ੍ਰਾਫਾਂ ਦੇ ਇੱਕ ਮਹਾਨ ਸਰੋਤ ਦੀ ਭਾਲ ਕਰ ਰਹੇ ਹੋ? ਤੁਸੀਂ ਸਹੀ ਜਗ੍ਹਾ ਤੇ ਆਏ ਹੋ. ਇੱਕ ਤਾਜ਼ਾ ਅਧਿਐਨ ਅਨੁਸਾਰ ਅੱਜ ਦੇ ਨੌਜਵਾਨਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਦੀ ਆਦਤ ਤੇਜ਼ੀ ਨਾਲ ਘਟ ਰਹੀ ਹੈ। ਉਹ ਕੁਝ ਸਕਿੰਟਾਂ ਤੋਂ ਵੱਧ ਸਮੇਂ ਲਈ ਦਿੱਤੇ ਗਏ ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਵਾਲੇ ਪੈਰੇ 'ਤੇ ਧਿਆਨ ਨਹੀਂ ਦੇ ਸਕਦੇ! ਨਾਲ ਹੀ, ਪੜ੍ਹਨਾ ਸਾਰੀਆਂ ਪ੍ਰਤੀਯੋਗੀ ਪ੍ਰੀਖਿਆਵਾਂ ਦਾ ਇੱਕ ਅਨਿੱਖੜਵਾਂ ਅੰਗ ਸੀ ਅਤੇ ਹੈ. ਇਸ ਲਈ, ਤੁਸੀਂ ਆਪਣੇ ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਨੂੰ ਕਿਵੇਂ ਸੁਧਾਰਦੇ ਹੋ? ਇਸ ਪ੍ਰਸ਼ਨ ਦਾ ਉੱਤਰ ਅਸਲ ਵਿੱਚ ਇੱਕ ਹੋਰ ਪ੍ਰਸ਼ਨ ਹੈ: ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਦੀ ਵਰਤੋਂ ਕੀ ਹੈ? ਪੜ੍ਹਨ ਦਾ ਮੁੱਖ ਉਦੇਸ਼ 'ਅਰਥ ਬਣਾਉਣਾ' ਹੈ.", output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਦੇ ਪੈਰਾਗ੍ਰਾਫਾਂ ਦੇ ਇੱਕ ਮਹਾਨ ਸਰੋਤ ਦੀ ਭਾਲ ਕਰ ਰਹੇ ਹੋ?]                                                                                                     |
|[ਤੁਸੀਂ ਸਹੀ ਜਗ੍ਹਾ ਤੇ ਆਏ ਹੋ.]                                                                                                                                            |
|[ਇੱਕ ਤਾਜ਼ਾ ਅਧਿਐਨ ਅਨੁਸਾਰ ਅੱਜ ਦੇ ਨੌਜਵਾਨਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਦੀ ਆਦਤ ਤੇਜ਼ੀ ਨਾਲ ਘਟ ਰਹੀ ਹੈ। ਉਹ ਕੁਝ ਸਕਿੰਟਾਂ ਤੋਂ ਵੱਧ ਸਮੇਂ ਲਈ ਦਿੱਤੇ ਗਏ ਅੰਗਰੇਜ਼ੀ ਪੜ੍ਹਨ ਵਾਲੇ ਪੈਰੇ 'ਤੇ ਧਿਆਨ ਨਹੀਂ ਦੇ ਸਕਦੇ!]|
|[ਨਾਲ ਹੀ, ਪੜ੍ਹਨਾ ਸਾਰੀਆਂ ਪ੍ਰਤੀਯੋਗੀ ਪ੍ਰੀਖਿਆਵਾਂ ਦਾ ਇੱਕ ਅਨਿੱਖੜਵਾਂ ਅੰਗ ਸੀ ਅਤੇ ਹੈ.]                                                                                           |
|[ਇਸ ਲਈ, ਤੁਸੀਂ ਆਪਣੇ ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਨੂੰ ਕਿਵੇਂ ਸੁਧਾਰਦੇ ਹੋ?]                                                                                                                |
|[ਇਸ ਪ੍ਰਸ਼ਨ ਦਾ ਉੱਤਰ ਅਸਲ ਵਿੱਚ ਇੱਕ ਹੋਰ ਪ੍ਰਸ਼ਨ ਹੈ:]                                                                                                                        |
|[ਪੜ੍ਹਨ ਦੇ ਹੁਨਰ ਦੀ ਵਰਤੋਂ ਕੀ ਹੈ?]                                                                                                                                        |
|[ਪੜ੍ਹਨ ਦਾ ਮੁੱਖ ਉਦੇਸ਼ 'ਅਰਥ ਬਣਾਉਣਾ' ਹੈ.]                                                                                                                                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+


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
|Language:|pa|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
