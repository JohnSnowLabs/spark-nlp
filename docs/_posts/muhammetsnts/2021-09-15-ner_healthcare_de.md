---
layout: model
title: Detect Symptoms, Treatments and Other Entities in German
author: John Snow Labs
name: ner_healthcare
date: 2021-09-15
tags: [ner, healthcare, licensed, de]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model can be used to detect clinical entities in medical text in German language.


## Predicted Entities


`DIAGLAB_PROCEDURE`, `MEDICAL_SPECIFICATION`, `MEDICAL_DEVICE`, `MEASUREMENT`, `BIOLOGICAL_CHEMISTRY`, `BODY_FLUID`, `TIME_INFORMATION`, `LOCAL_SPECIFICATION`, `BIOLOGICAL_PARAMETER`, `PROCESS`, `MEDICATION`, `DOSING`, `DEGREE`, `MEDICAL_CONDITION`, `PERSON`, `TISSUE`, `STATE_OF_HEALTH`, `BODY_PART`, `TREATMENT`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_3.0.0_3.0_1631687601139.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_3.0.0_3.0_1631687601139.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
document_assembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")\
   .setInputCols(["sentence","token"])\
   .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_healthcare", "de", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

clinical_ner_converter = NerConverter() \
	.setInputCols(["sentence", "token", "ner"]) \
	.setOutputCol("ner_chunk")

...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, clinical_ner_converter])

data = spark.createDataFrame([["Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt."]]).toDF("text")

result = nlp_pipeline.fit(data).transform(data)
```
```scala
...
val document_assembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols(Array("sentence"))
		.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de", "clinical/models")
   .setInputCols(["sentence", "token"])
   .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_healthcare", "de", "clinical/models") 
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")

val clinical_ner_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "ner"))
		.setOutputCol("ner_chunk")

...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, clinical_ner_converter))

val data = Seq("""Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+-----------------+---------------------+
|chunk            |label                |
+-----------------+---------------------+
|Kleinzellige     |MEASUREMENT          |
|Bronchialkarzinom|MEDICAL_CONDITION    |
|Kleinzelliger    |MEDICAL_SPECIFICATION|
|Lungenkrebs      |MEDICAL_CONDITION    |
|SCLC             |MEDICAL_CONDITION    |
|Hernia           |MEDICAL_CONDITION    |
|femoralis        |LOCAL_SPECIFICATION  |
|Akne             |MEDICAL_CONDITION    |
|einseitig        |MEASUREMENT          |
|hochmalignes     |MEDICAL_CONDITION    |
|bronchogenes     |BODY_PART            |
|Karzinom         |MEDICAL_CONDITION    |
|Lunge            |BODY_PART            |
|Hauptbronchus    |BODY_PART            |
|mittlere         |MEASUREMENT          |
|Prävalenz        |MEDICAL_CONDITION    |
+-----------------+---------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_healthcare|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|


## Benchmarking


```bash
                label       tp      fp      fn    total  precision  recall      f1
 BIOLOGICAL_PARAMETER   1186.0   651.0   429.0   1615.0     0.6456  0.7344  0.6871
           BODY_FLUID     32.0     8.0    23.0     55.0        0.8  0.5818  0.6737
               PERSON   3927.0   293.0   641.0   4568.0     0.9306  0.8597  0.8937
               DOSING    203.0    96.0   155.0    358.0     0.6789   0.567   0.618
    DIAGLAB_PROCEDURE   2373.0   812.0   873.0   3246.0     0.7451  0.7311   0.738
               TISSUE      4.0     3.0     2.0      6.0     0.5714  0.6667  0.6154
            BODY_PART   1859.0   513.0   384.0   2243.0     0.7837  0.8288  0.8056
           MEDICATION   3307.0   925.0  1075.0   4382.0     0.7814  0.7547  0.7678
      STATE_OF_HEALTH    602.0   131.0   162.0    764.0     0.8213   0.788  0.8043
  LOCAL_SPECIFICATION    231.0    86.0    97.0    328.0     0.7287  0.7043  0.7163
          MEASUREMENT   6472.0  1612.0  1691.0   8163.0     0.8006  0.7928  0.7967
            TREATMENT   9262.0  2054.0  2380.0  11642.0     0.8185  0.7956  0.8069
MEDICAL_SPECIFICATION   1455.0   782.0   493.0   1948.0     0.6504  0.7469  0.6953
    MEDICAL_CONDITION  10464.0  2243.0  2364.0  12828.0     0.8235  0.8157  0.8196
     TIME_INFORMATION   1496.0   534.0   603.0   2099.0     0.7369  0.7127  0.7246
              PROCESS    526.0   232.0   251.0    777.0     0.6939   0.677  0.6853
 BIOLOGICAL_CHEMISTRY    524.0   261.0   392.0    916.0     0.6675  0.5721  0.6161
```
