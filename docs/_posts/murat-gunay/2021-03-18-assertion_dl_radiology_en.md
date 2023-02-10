---
layout: model
title: Detect Assertion Status for Radiology
author: John Snow Labs
name: assertion_dl_radiology
date: 2021-03-18
tags: [assertion, en, licensed, radiology, clinical]
task: Assertion Status
language: en
edition: Healthcare NLP 2.7.4
spark_version: 2.4
supported: true
annotator: AssertionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to clinical entities extracted by Radiology NER based on their context in the text.

## Predicted Entities

`Confirmed`, `Suspected`, `Negative`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION_RADIOLOGY/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_radiology_en_2.7.4_2.4_1616071311532.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_radiology_en_2.7.4_2.4_1616071311532.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Extract radiology entities using the radiology NER model in the pipeline and assign assertion status for them with `assertion_dl_radiology` pretrained model. Note: Example for demo purpose taken from: https://www.mtsamples.com/site/pages/sample.asp?Type=95-Radiology&Sample=1391-Chest%20PA%20&%20Lateral


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

radiology_ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(["ImagingFindings"])

radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = LightPipeline(nlpPipeline.fit(empty_data))

text = """
INTERPRETATION: There has been interval development of a moderate left-sided pneumothorax with near complete collapse of the left upper lobe. The lower lobe appears aerated. There is stable, diffuse, bilateral interstitial thickening with no definite acute air space consolidation. The heart and pulmonary vascularity are within normal limits. Left-sided port is seen with Groshong tip at the SVC/RA junction. No evidence for acute fracture, malalignment, or dislocation."""

result = model.fullAnnotate(text)

```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val radiology_ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")
    .setWhiteList(Array("ImagingFindings"))

val radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val nlpPipeline = new Pipeline().setStages(Array(documentAssembler,  sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion))

text = """
INTERPRETATION: There has been interval development of a moderate left-sided pneumothorax with near complete collapse of the left upper lobe. The lower lobe appears aerated. There is stable, diffuse, bilateral interstitial thickening with no definite acute air space consolidation. The heart and pulmonary vascularity are within normal limits. Left-sided port is seen with Groshong tip at the SVC/RA junction. No evidence for acute fracture, malalignment, or dislocation."""

val data = Seq("text").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.assert.radiology").predict("""
INTERPRETATION: There has been interval development of a moderate left-sided pneumothorax with near complete collapse of the left upper lobe. The lower lobe appears aerated. There is stable, diffuse, bilateral interstitial thickening with no definite acute air space consolidation. The heart and pulmonary vascularity are within normal limits. Left-sided port is seen with Groshong tip at the SVC/RA junction. No evidence for acute fracture, malalignment, or dislocation.""")
```

</div>

## Results

```bash
|    | ner_chunk                     | assertion   |
|---:|:------------------------------|:------------|
|  0 | pneumothorax                  | Confirmed   |
|  1 | complete collapse             | Confirmed   |
|  2 | aerated                       | Confirmed   |
|  3 | thickening                    | Confirmed   |
|  4 | acute air space consolidation | Negative    |
|  5 | within normal limits          | Confirmed   |
|  6 | acute fracture                | Negative    |
|  7 | malalignment                  | Negative    |
|  8 | dislocation                   | Negative    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_radiology|
|Compatibility:|Healthcare NLP 2.7.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Custom internal labeled radiology dataset.

## Benchmarking

```bash
label	       tp	 fp	     fn	     prec	     rec	     f1
Suspected	   629	 155	 159	 0.8022959	 0.7982234	 0.80025446
Negative	   417	 53	     36	     0.88723403	 0.9205298	 0.9035753
Confirmed	   2252	 173	 186	 0.9286598	 0.92370796	 0.92617726
tp: 3298 fp: 381 fn: 381 labels: 3
Macro-average	 prec: 0.87272996, rec: 0.88082033, f1: 0.8767565
Micro-average	 prec: 0.89643925, rec: 0.89643925, f1: 0.89643925
```