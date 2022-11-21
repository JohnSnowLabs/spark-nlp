---
layout: model
title: Detect Problems, Tests and Treatments (ner_clinical_large)
author: John Snow Labs
name: ner_clinical
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for clinical terms. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities

`PROBLEM`, `TEST`, `TREATMENT`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_en_3.0.0_3.0_1617208419368.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_large", "en", "clinical/models") \
        .setInputCols(["sentence", "token", "embeddings"]) \
        .setOutputCol("ner")

ner_converter = NerConverter()\
 	    .setInputCols(["sentence", "token", "ner"])\
 	    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.']], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
        .setInputCols("document")
        .setOutputCol("sentence")

val tokenizer = new Tokenizer()
        .setInputCols("sentence")
        .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_clinical_large", "en", "clinical/models")
        .setInputCols(Array("sentence", "token", "embeddings"))
        .setOutputCol("ner")

val ner_converter = new NerConverter()
 	    .setInputCols(Array("sentence", "token", "ner"))
 	    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.clinical").predict("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.""")
```

</div>

## Results

```bash
+-----------------------------------------------------------+---------+
|chunk                                                      |ner_label|
+-----------------------------------------------------------+---------+
|the G-protein-activated inwardly rectifying potassium (GIRK|TREATMENT|
|the genomicorganization                                    |TREATMENT|
|a candidate gene forType II diabetes mellitus              |PROBLEM  |
|byapproximately                                            |TREATMENT|
|single nucleotide polymorphisms                            |TREATMENT|
|aVal366Ala substitution                                    |TREATMENT|
|an 8 base-pair                                             |TREATMENT|
|insertion/deletion                                         |PROBLEM  |
|Ourexpression studies                                      |TEST     |
|the transcript in various humantissues                     |PROBLEM  |
|fat andskeletal muscle                                     |PROBLEM  |
|furtherstudies                                             |PROBLEM  |
|the KCNJ9 protein                                          |TREATMENT|
|evaluation                                                 |TEST     |
|Type II diabetes                                           |PROBLEM  |
|the treatment                                              |TREATMENT|
|breast cancer                                              |PROBLEM  |
|the standard therapy                                       |TREATMENT|
|anthracyclines                                             |TREATMENT|
|taxanes                                                    |TREATMENT|
+-----------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on augmented 2010 i2b2 challenge data with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking

```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-TREATMENT   |  6625 |  1187 |  1329 | 0.848054 | 0.832914 | 0.840416 |
|  1 | I-PROBLEM     | 15142 |  1976 |  2542 | 0.884566 | 0.856254 | 0.87018  |
|  2 | B-PROBLEM     | 11005 |  1065 |  1587 | 0.911765 | 0.873968 | 0.892466 |
|  3 | I-TEST        |  6748 |   923 |  1264 | 0.879677 | 0.842237 | 0.86055  |
|  4 | B-TEST        |  8196 |   942 |  1029 | 0.896914 | 0.888455 | 0.892665 |
|  5 | B-TREATMENT   |  8271 |  1265 |  1073 | 0.867345 | 0.885167 | 0.876165 |
|  6 | Macro-average | 55987 |  7358 |  8824 | 0.881387 | 0.863166 | 0.872181 |
|  7 | Micro-average | 55987 |  7358 |  8824 | 0.883842 | 0.86385  | 0.873732 |
```