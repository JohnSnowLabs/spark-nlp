---
layout: model
title: Detect Assertion Status (assertion_dl) - supports confidence scores
author: John Snow Labs
name: assertion_dl
date: 2021-01-26
task: Assertion Status
language: en
edition: Healthcare NLP 2.7.2
spark_version: 2.4
tags: [assertion, en, licensed, clinical]
supported: true
annotator: AssertionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to clinical entities extracted by NER based on their context in the text.

## Predicted Entities

`absent`, `present`, `conditional`, `associated_with_someone_else`, `hypothetical`, `possible`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.7.2_2.4_1611647201607.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.7.2_2.4_1611647201607.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
documentAssembler, 
sentenceDetector,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
clinical_assertion
])

data = spark.createDataFrame([["The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."]]).toDF("text")

model = nlpPipeline.fit(data)

result = model.transform(data)
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

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val nlpPipeline = new Pipeline().setStages(Array(
documentAssembler, 
sentenceDetector,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
clinical_assertion
))

val text = """The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."""


val data = Seq(text).toDS.toDF("text")

val results = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.assert").predict("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.""")
```

</div>

## Results

```bash
+--------------------+---------+---------+
|               chunk|ner_label|assertion|
+--------------------+---------+---------+
|                 Kir|     TEST|  present|
|               GIRK3|     TEST|  present|
|  chromosome 1q21-23|TREATMENT|  present|
|a candidate gene ...|  PROBLEM| possible|
|        coding exons|     TEST|   absent|
|     byapproximately|     TEST|  present|
|             introns|     TEST|   absent|
|single nucleotide...|  PROBLEM|  present|
|aVal366Ala substi...|TREATMENT|  present|
|      an 8 base-pair|  PROBLEM|  present|
|                 bp)|     TEST|  present|
|Ourexpression stu...|     TEST|  present|
|The characterizat...|     TEST|  present|
|      furtherstudies|TREATMENT|  present|
|       KCNJ9 protein|     TEST|  present|
|          evaluation|     TEST|  present|
+--------------------+---------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Trained on i2b2 assertion data

## Benchmarking

```bash
|    | label                        |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|:-----------------------------|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | absent                       |   791 |   47 |   80 | 0.943914 | 0.908152 | 0.925688 |
|  1 | present                      |  2499 |  169 |  120 | 0.936657 | 0.954181 | 0.945338 |
|  2 | conditional                  |    23 |   19 |   21 | 0.547619 | 0.522727 | 0.534884 |
|  3 | associated_with_someone_else |    38 |    2 |   11 | 0.95     | 0.77551  | 0.853933 |
|  4 | hypothetical                 |   163 |   19 |   21 | 0.895604 | 0.88587  | 0.89071  |
|  5 | possible                     |   119 |   61 |   64 | 0.661111 | 0.650273 | 0.655647 |
|  6 | Macro-average                | 3633  | 317  |  317 | 0.822484 | 0.782786 | 0.802144 |
|  7 | Micro-average                | 3633  | 317  |  317 | 0.919747 | 0.919747 | 0.919747 |

```