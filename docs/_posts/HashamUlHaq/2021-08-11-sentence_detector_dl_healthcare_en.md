---
layout: model
title: Detect Sentences in Healthcare Texts
author: John Snow Labs
name: sentence_detector_dl_healthcare
date: 2021-08-11
tags: [licensed, clinical, en, sentence_detection]
task: Sentence Detection
language: en
edition: Spark NLP for Healthcare 3.2.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

In this model, we treated the sentence boundary detection task as a classification problem based on a paper {Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed) using CNN architecture. We also modified the original implemenation a little bit to cover broken sentences and some impossible end of line chars.

## Predicted Entities

Breaks text in sentences.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/SENTENCE_DETECTOR_HC/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sentence_detector_dl_healthcare_en_3.2.0_3.0_1628678815210.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

text = """He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.
Size: Prostate gland measures 10x1.1x 4.9 cm (LS x AP x TS). Estimated volume is 
51.9 ml. , and is mildly enlarged in size.Normal delineation pattern of the prostate gland is preserved.
"""

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

result = sd_model.fullAnnotate(text)


```
```scala
val documenter = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val text = "He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.
Size: Prostate gland measures 10x1.1x 4.9 cm (LS x AP x TS). Estimated volume is 
51.9 ml. , and is mildly enlarged in size.Normal delineation pattern of the prostate gland is preserved."

val data = Seq(text).toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | sentences                                                                                                                                                                                                              |
|---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety. |
|  1 | Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.                                                                                                                                                   |
|  2 | Size: Prostate gland measures 10x1.1x 4.9 cm (LS x AP x TS).                                                                                                                                                           |
|  3 | Estimated volume is                                                                                                                                                                                                    |
|    | 51.9 ml. , and is mildly enlarged in size.                                                                                                                                                                             |
|  4 | Normal delineation pattern of the prostate gland is preserved.                                                                                                                                                         |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl_healthcare|
|Compatibility:|Spark NLP for Healthcare 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|en|

## Data Source

Healthcare SDDL model is trained on in-house domain specific data.

## Benchmarking

```bash
|    | Accuracy | Recall   | Prec     | F1   |
|---:|---------:|---------:|---------:|-----:|
|  0 | 0.98     | 1.00     | 0.96     | 0.98 |
```