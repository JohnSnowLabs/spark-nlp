---
layout: model
title: Sentence Detection in Yiddish Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [open_source, sentence_detection, yi]
task: Sentence Detection
language: yi
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_yi_3.2.0_3.0_1630323089681.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_yi_3.2.0_3.0_1630323089681.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "yi") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""איר זוכט פֿאַר אַ גרויס מקור פון לייענען פּאַראַגראַפס אין ענגליש? איר'ווע קומען צו די רעכט אָרט. לויט צו אַ פריש לערנען, די מידע פון לייענען אין הייַנט ס יוגנט איז ראַפּאַדלי דיקריסינג. זיי קענען נישט פאָקוס אויף אַ געגעבן פּאַראַגראַף פֿאַר ענגליש לייענען פֿאַר מער ווי אַ ביסל סעקונדעס! לייענען איז געווען און איז אַ ינטאַגראַל טייל פון אַלע קאַמפּעטיטיוו יגזאַמז. אַזוי ווי טאָן איר פֿאַרבעסערן דיין לייענען סקילז? דער ענטפער צו דעם קשיא איז אַקשלי אן אנדער קשיא: וואָס איז די נוצן פון לייענען סקילז? דער הויפּט ציל פון לייענען איז 'צו מאַכן זינען'.""")



```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "yi")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("איר זוכט פֿאַר אַ גרויס מקור פון לייענען פּאַראַגראַפס אין ענגליש? איר'ווע קומען צו די רעכט אָרט. לויט צו אַ פריש לערנען, די מידע פון לייענען אין הייַנט ס יוגנט איז ראַפּאַדלי דיקריסינג. זיי קענען נישט פאָקוס אויף אַ געגעבן פּאַראַגראַף פֿאַר ענגליש לייענען פֿאַר מער ווי אַ ביסל סעקונדעס! לייענען איז געווען און איז אַ ינטאַגראַל טייל פון אַלע קאַמפּעטיטיוו יגזאַמז. אַזוי ווי טאָן איר פֿאַרבעסערן דיין לייענען סקילז? דער ענטפער צו דעם קשיא איז אַקשלי אן אנדער קשיא: וואָס איז די נוצן פון לייענען סקילז? דער הויפּט ציל פון לייענען איז 'צו מאַכן זינען'.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
nlu.load('yi.sentence_detector').predict("איר זוכט פֿאַר אַ גרויס מקור פון לייענען פּאַראַגראַפס אין ענגליש? איר'ווע קומען צו די רעכט אָרט. לויט צו אַ פריש לערנען, די מידע פון לייענען אין הייַנט ס יוגנט איז ראַפּאַדלי דיקריסינג. זיי קענען נישט פאָקוס אויף אַ געגעבן פּאַראַגראַף פֿאַר ענגליש לייענען פֿאַר מער ווי אַ ביסל סעקונדעס! לייענען איז געווען און איז אַ ינטאַגראַל טייל פון אַלע קאַמפּעטיטיוו יגזאַמז. אַזוי ווי טאָן איר פֿאַרבעסערן דיין לייענען סקילז? דער ענטפער צו דעם קשיא איז אַקשלי אן אנדער קשיא: וואָס איז די נוצן פון לייענען סקילז? דער הויפּט ציל פון לייענען איז 'צו מאַכן זינען'.", output_level ='sentence')  
```
</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------+
|result                                                                                                  |
+--------------------------------------------------------------------------------------------------------+
|[איר זוכט פֿאַר אַ גרויס מקור פון לייענען פּאַראַגראַפס אין ענגליש?]                                    				 |
|[איר'ווע קומען צו די רעכט אָרט.]                                                                          		 |
|[לויט צו אַ פריש לערנען, די מידע פון לייענען אין הייַנט ס יוגנט איז ראַפּאַדלי דיקריסינג.]              					 |
|[זיי קענען נישט פאָקוס אויף אַ געגעבן פּאַראַגראַף פֿאַר ענגליש לייענען פֿאַר מער ווי אַ ביסל סעקונדעס!]						 |
|[לייענען איז געווען און איז אַ ינטאַגראַל טייל פון אַלע קאַמפּעטיטיוו יגזאַמז.]                         					 |
|[אַזוי ווי טאָן איר פֿאַרבעסערן דיין לייענען סקילז?]                                                   			 |
|[דער ענטפער צו דעם קשיא איז אַקשלי אן אנדער קשיא:]                                                     		 |
|[וואָס איז די נוצן פון לייענען סקילז?]                                                                 			 |
|[דער הויפּט ציל פון לייענען איז 'צו מאַכן זינען'.]                                                     			 |
+--------------------------------------------------------------------------------------------------------+

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
|Language:|yi|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
