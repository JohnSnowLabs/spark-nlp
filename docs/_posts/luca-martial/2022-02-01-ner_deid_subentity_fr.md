---
layout: model
title: Detect PHI for Deidentification purposes (French)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-02-01
tags: [deid, fr, licensed]
task: De-identification
language: fr
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.

Deidentification NER (French) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 13 entities. This NER model is trained with a custom dataset internally annotated, a [public dataset of French company names](https://www.data.gouv.fr/fr/datasets/entreprises-immatriculees-en-2017/), a [public dataset of French hospital names](https://salesdorado.com/fichiers-prospection/hopitaux/) and several data augmentation mechanisms.

## Predicted Entities

`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `LOCATION`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_fr_3.4.0_3.0_1643749771101.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("word2vec_wac_200", "fr")\
	.setInputCols(["sentence", "token"])\
	.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models")\
        .setInputCols(["sentence","token", "word_embeddings"])\
        .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_ner])

text = ["Il s'agit d'un homme agé de 49 ans adressé au CHU de Montpellier pour un diabète mal contrôlé avec des symptômes datant de Mars 2015."]

df = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
        .setInputCols("document")
        .setOutputCol("sentence")

val tokenizer = Tokenizer()
        .setInputCols("sentence")
        .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("word2vec_wac_200", "fr")
    .setInputCols("sentence", "token")
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models")
        .setInputCols("sentence","token","embeddings")
        .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))

val text = "Il s'agit d'un homme agé de 49 ans adressé au CHU de Montpellier pour un diabète mal contrôlé avec des symptômes datant de Mars 2015."

val df = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------+----------+
|      token| ner_label|
+-----------+----------+
|         Il|         O|
|     s'agit|         O|
|       d'un|         O|
|      homme|         O|
|        agé|     B-AGE|
|         de|     I-AGE|
|         49|     I-AGE|
|        ans|     I-AGE|
|    adressé|         O|
|         au|         O|
|        CHU|B-HOSPITAL|
|         de|I-HOSPITAL|
|Montpellier|I-HOSPITAL|
|       pour|         O|
|         un|         O|
|    diabète|         O|
|        mal|         O|
|   contrôlé|         O|
|       avec|         O|
|        des|         O|
|  symptômes|         O|
|     datant|         O|
|         de|         O|
|       Mars|    B-DATE|
|       2015|    I-DATE|
|          .|         O|
+-----------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|14.7 MB|

## References

- Internal JSL annotated corpus
- [Public dataset of French company names](https://www.data.gouv.fr/fr/datasets/entreprises-immatriculees-en-2017/)
- [Public dataset of French hospital names](https://salesdorado.com/fichiers-prospection/hopitaux/)

## Benchmarking

```bash
+-------------+------+----+----+------+---------+------+------+
|       entity|    tp|  fp|  fn| total|precision|recall|    f1|
+-------------+------+----+----+------+---------+------+------+
|      PATIENT| 230.0| 7.0| 9.0| 239.0|   0.9705|0.9623|0.9664|
|     HOSPITAL| 239.0| 8.0| 6.0| 245.0|   0.9676|0.9755|0.9715|
|         DATE|1165.0|15.0| 7.0|1172.0|   0.9873| 0.994|0.9906|
| ORGANIZATION|  76.0|18.0|25.0| 101.0|   0.8085|0.7525|0.7795|
|         MAIL|  68.0| 0.0| 0.0|  68.0|      1.0|   1.0|   1.0|
|     USERNAME|  46.0| 1.0| 6.0|  52.0|   0.9787|0.8846|0.9293|
|     LOCATION| 187.0|13.0|22.0| 209.0|    0.935|0.8947|0.9144|
|          ZIP|  33.0| 1.0| 6.0|  39.0|   0.9706|0.8462|0.9041|
|MEDICALRECORD| 101.0| 4.0| 7.0| 108.0|   0.9619|0.9352|0.9484|
|   PROFESSION| 285.0|35.0|46.0| 331.0|   0.8906| 0.861|0.8756|
|        PHONE|  65.0| 2.0| 2.0|  67.0|   0.9701|0.9701|0.9701|
|       DOCTOR| 485.0| 5.0| 2.0| 487.0|   0.9898|0.9959|0.9928|
|          AGE| 277.0|40.0|39.0| 316.0|   0.8738|0.8766|0.8752|
+-------------+------+----+----+------+---------+------+------+

+-----------------+
|            macro|
+-----------------+
|0.932154418042752|
+-----------------+

+------------------+
|             micro|
+------------------+
|0.9518841684676022|
+------------------+

```
