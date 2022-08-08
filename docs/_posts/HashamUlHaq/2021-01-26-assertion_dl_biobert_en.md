---
layout: model
title: Detect Assertion Status (assertion_dl_biobert) - supports confidence scores
author: John Snow Labs
name: assertion_dl_biobert
date: 2021-01-26
task: Assertion Status
language: en
edition: Spark NLP for Healthcare 2.7.2
spark_version: 2.4
tags: [assertion, en, licensed, clinical]
supported: true
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_biobert_en_2.7.2_2.4_1611647486798.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

word_embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
.setInputCols(["document",'token'])\
.setOutputCol("embeddings")

clinical_ner = NerDLModel.pretrained("ner_clinical_biobert", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter() \
.setInputCols(["sentence", "token", "ner"]) \
.setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl_biobert", "en", "clinical/models") \
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



{:.nlu-block}
```python
import nlu
nlu.load("en.assert.biobert").predict("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.""")
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
|Model Name:|assertion_dl_biobert|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Trained on i2b2 assertion data.

## Benchmarking

```bash
|    | label                        |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|:-----------------------------|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | absent                       |   769 |   51 |   57 | 0.937805 | 0.930993 | 0.934386 |
|  1 | present                      |  2575 |  161 |  102 | 0.941155 | 0.961898 | 0.951413 |
|  2 | conditional                  |    20 |   14 |   23 | 0.588235 | 0.465116 | 0.519481 |
|  3 | associated_with_someone_else |    51 |    9 |   15 | 0.85     | 0.772727 | 0.809524 |
|  4 | hypothetical                 |   129 |   13 |   15 | 0.908451 | 0.895833 | 0.902098 |
|  5 | possible                     |   114 |   44 |   80 | 0.721519 | 0.587629 | 0.647727 |
|  6 | Macro-average                | 3658  | 292  |  292 | 0.824527 | 0.769033 | 0.795814 |
|  7 | Micro-average                | 3658  | 292  |  292 | 0.926076 | 0.926076 | 0.926076 |

```