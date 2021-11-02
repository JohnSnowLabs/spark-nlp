---
layout: model
title: Detect Radiology Concepts (WIP)
author: John Snow Labs
name: jsl_rd_ner_wip_greedy_clinical
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract clinical entities from Radiology reports using pretrained NER model.

## Predicted Entities

`Score`, `Test_Result`, `Medical_Device`, `Units`, `Imaging_Technique`, `Direction`, `ImagingTest`, `ManualFix`, `Symptom`, `ImagingFindings`, `OtherFindings`, `Test`, `Measurements`, `Procedure`, `Disease_Syndrome_Disorder`, `BodyPart`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_clinical_en_3.0.0_3.0_1617260438155.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_clinical", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[String]).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_rd_ner_wip_greedy_clinical|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Benchmarking

```bash
label                       tp    fp    fn    prec        rec         f1
B-Units                     350   11    12    0.9695291   0.9668508   0.96818805
B-Medical_Device            504   126   97    0.8         0.8386023   0.8188465
B-BodyPart                  3416  399   288   0.89541286  0.9222462   0.90863144
I-BodyPart                  812   111   186   0.87974     0.81362724  0.84539306
B-Imaging_Technique         333   53    45    0.8626943   0.88095236  0.87172776
B-Procedure                 368   128   114   0.7419355   0.7634855   0.75255626
B-Direction                 1945  147   137   0.9297323   0.9341979   0.93195975
I-ImagingTest               36    17    29    0.6792453   0.5538462   0.6101695
I-ManualFix                 3     0     3     1.0         0.5         0.6666667
I-Test_Result               2     8     0     0.2         1.0         0.3333333
B-Measurements              545   25    33    0.95614034  0.94290656  0.9494774
B-OtherFindings             15    11    85    0.5769231   0.15        0.23809524
B-ImagingFindings           2482  600   612   0.8053212   0.8021978   0.8037565
I-Units                     0     0     1     0.0         0.0         0.0
B-Test_Result               10    8     26    0.5555556   0.2777778   0.3703704
B-Test                      229   46    40    0.83272725  0.85130113  0.84191173
B-Score                     1     0     0     1.0         1.0         1.0
I-OtherFindings             6     10    61    0.375       0.08955224  0.14457832
B-ManualFix                 24    1     0     0.96        1.0         0.9795918
I-Procedure                 205   107   110   0.65705127  0.6507937   0.65390754
I-Imaging_Technique         140   45    63    0.7567568   0.6896552   0.7216495
I-Measurements              64    4     6     0.9411765   0.9142857   0.92753625
B-ImagingTest               514   117   59    0.81458     0.89703315  0.85382056
I-Test                      76    20    18    0.7916667   0.80851066  0.8000001
I-Symptom                   157   80    117   0.6624473   0.5729927   0.61448133
I-ImagingFindings           1816  586   727   0.75603664  0.71411717  0.73447925
B-Disease_Syndrome_Disorder 1339  297   245   0.81845963  0.8453283   0.831677
B-Symptom                   594   157   158   0.7909454   0.7898936   0.79041916
I-Disease_Syndrome_Disorder 433   241   160   0.6424332   0.7301855   0.6835044
I-Medical_Device            481   88    78    0.8453427   0.8604651   0.8528369
I-Direction                 373   21    48    0.9467005   0.88598573  0.91533744

tp: 17273 fp: 3464 fn: 3558 labels: 31
Macro-average prec: 0.75624377, rec: 0.7305416, f1: 0.7431705
Micro-average prec: 0.8329556, rec: 0.8291969, f1: 0.831072

```