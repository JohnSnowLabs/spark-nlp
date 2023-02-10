---
layout: model
title: Sentence Detection in English Texts
author: John Snow Labs
name: sentence_detector_dl
date: 2021-01-02
task: Sentence Detection
language: en
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [en, sentence_detection, open_source]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTENCE_DETECTOR/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/20.SentenceDetectorDL_Healthcare.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_en_2.7.0_2.4_1609611052663.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_en_2.7.0_2.4_1609611052663.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "en") \
.setInputCols(["document"]) \
.setOutputCol("sentences")
sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved.""")
```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")
val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("sentence_detector").predict("""John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved.""")
```

</div>

## Results

```bash
+---+------------------------------+
| 0 | John loves Mary.             |
+---+------------------------------+
| 1 | Mary loves Peter             |
+---+------------------------------+
| 2 | Peter loves Helen .          |
+---+------------------------------+
| 3 | Helen loves John;            |
+---+------------------------------+
| 4 | Total: four people involved. |
+---+------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|en|

## Data Source

Please visit the repo for more information https://github.com/dbmdz/deep-eos

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```