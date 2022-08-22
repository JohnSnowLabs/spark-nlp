---
layout: model
title: Sentence Detection in Telugu Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [te, sentence_detection, open_source]
task: Embeddings
language: te
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
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_te_3.2.0_3.0_1630338728542.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "te") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు. ఇటీవలి అధ్యయనం ప్రకారం, నేటి యువతలో చదివే అలవాటు వేగంగా తగ్గుతోంది. వారు కొన్ని సెకన్ల కంటే ఎక్కువ ఇచ్చిన ఆంగ్ల పఠన పేరాపై దృష్టి పెట్టలేరు! అలాగే, చదవడం అనేది అన్ని పోటీ పరీక్షలలో అంతర్భాగం. కాబట్టి, మీరు మీ పఠన నైపుణ్యాలను ఎలా మెరుగుపరుచుకుంటారు? ఈ ప్రశ్నకు సమాధానం నిజానికి మరొక ప్రశ్న: పఠన నైపుణ్యాల ఉపయోగం ఏమిటి? చదవడం యొక్క ముఖ్య ఉద్దేశ్యం 'అర్థం చేసుకోవడం'.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "te")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు. ఇటీవలి అధ్యయనం ప్రకారం, నేటి యువతలో చదివే అలవాటు వేగంగా తగ్గుతోంది. వారు కొన్ని సెకన్ల కంటే ఎక్కువ ఇచ్చిన ఆంగ్ల పఠన పేరాపై దృష్టి పెట్టలేరు! అలాగే, చదవడం అనేది అన్ని పోటీ పరీక్షలలో అంతర్భాగం. కాబట్టి, మీరు మీ పఠన నైపుణ్యాలను ఎలా మెరుగుపరుచుకుంటారు? ఈ ప్రశ్నకు సమాధానం నిజానికి మరొక ప్రశ్న: పఠన నైపుణ్యాల ఉపయోగం ఏమిటి? చదవడం యొక్క ముఖ్య ఉద్దేశ్యం 'అర్థం చేసుకోవడం'.").toDF("text")

val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
import nlu

nlu.load('te.sentence_detector').predict("ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు. ఇటీవలి అధ్యయనం ప్రకారం, నేటి యువతలో చదివే అలవాటు వేగంగా తగ్గుతోంది. వారు కొన్ని సెకన్ల కంటే ఎక్కువ ఇచ్చిన ఆంగ్ల పఠన పేరాపై దృష్టి పెట్టలేరు! అలాగే, చదవడం అనేది అన్ని పోటీ పరీక్షలలో అంతర్భాగం. కాబట్టి, మీరు మీ పఠన నైపుణ్యాలను ఎలా మెరుగుపరుచుకుంటారు? ఈ ప్రశ్నకు సమాధానం నిజానికి మరొక ప్రశ్న: పఠన నైపుణ్యాల ఉపయోగం ఏమిటి? చదవడం యొక్క ముఖ్య ఉద్దేశ్యం 'అర్థం చేసుకోవడం'.", output_level ='sentence')  

```
</div>

## Results

```bash
+--------------------------------------------------------------------------+
|result                                                                    |
+--------------------------------------------------------------------------+
|[ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా?]                     |
|[మీరు సరైన స్థలానికి వచ్చారు.]                                            |
|[ఇటీవలి అధ్యయనం ప్రకారం, నేటి యువతలో చదివే అలవాటు వేగంగా తగ్గుతోంది.]     |
|[వారు కొన్ని సెకన్ల కంటే ఎక్కువ ఇచ్చిన ఆంగ్ల పఠన పేరాపై దృష్టి పెట్టలేరు!]|
|[అలాగే, చదవడం అనేది అన్ని పోటీ పరీక్షలలో అంతర్భాగం.]                      |
|[కాబట్టి, మీరు మీ పఠన నైపుణ్యాలను ఎలా మెరుగుపరుచుకుంటారు?]                |
|[ఈ ప్రశ్నకు సమాధానం నిజానికి మరొక ప్రశ్న:]                                |
|[పఠన నైపుణ్యాల ఉపయోగం ఏమిటి?]                                             |
|[చదవడం యొక్క ముఖ్య ఉద్దేశ్యం 'అర్థం చేసుకోవడం'.]                          |
+--------------------------------------------------------------------------+


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
|Language:|te|

## Benchmarking

```bash
Accuracy:      0.98
Recall:        1.00
Precision:     0.96
F1:            0.98
```