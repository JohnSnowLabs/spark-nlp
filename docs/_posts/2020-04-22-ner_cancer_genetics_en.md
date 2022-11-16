---
layout: model
title: Detect Genetic Cancer Entities
author: John Snow Labs
name: ner_cancer_genetics
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-22
task: Named Entity Recognition
edition: Healthcare NLP 2.4.2
spark_version: 2.4
tags: [clinical,licensed,ner,en]
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for biology and genetics terms.

## Predicted Entities 
``DNA``, ``RNA``, ``cell_line``, ``cell_type``, ``protein``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cancer_genetics_en_2.4.2_2.4_1587567870408.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_cancer_genetics", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.']], ["text"]))
```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_cancer_genetics", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.cancer").predict("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.""")
```

</div>

{:.h2_title}
## Results

```bash
+-------------------+---------+
|              token|ner_label|
+-------------------+---------+
|                The|        O|
|              human|B-protein|
|              KCNJ9|I-protein|
|                  (|        O|
|                Kir|B-protein|
|                3.3|I-protein|
|                  ,|        O|
|              GIRK3|B-protein|
|                  )|        O|
|                 is|        O|
|                  a|        O|
|             member|        O|
|                 of|        O|
|                the|        O|
|G-protein-activated|B-protein|
|           inwardly|I-protein|
|         rectifying|I-protein|
|          potassium|I-protein|
|                  (|I-protein|
|               GIRK|I-protein|
|                  )|I-protein|
|            channel|I-protein|
|             family|I-protein|
|                  .|        O|
|               Here|        O|
|                 we|        O|
|           describe|        O|
|                the|        O|
|genomicorganization|        O|
|                 of|        O|
|                the|        O|
|              KCNJ9|    B-DNA|
|              locus|    I-DNA|
|                 on|        O|
|         chromosome|    B-DNA|
|            1q21-23|    I-DNA|
+-------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| Name:          | ner_cancer_genetics              |
| Type:   | NerDLModel                       |
| Compatibility: | 2.4.2                            |
| License:       | Licensed                         |
| Edition:       | Official                       |
|Input labels:        | sentence, token, word_embeddings |
|Output labels:       | ner                              |
| Language:      | en                               |
| Dependencies: | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013 with `embeddings_clinical`.
https://aclanthology.org/W13-2008/

{:.h2_title}
## Benchmarking
```bash
label        tp    fp   fn   prec         rec          f1

B-cell_line  581   148  151  0.79698217   0.79371583   0.79534566
I-DNA        2751  735  317  0.7891566    0.89667535   0.8394873
I-protein    4416  867  565  0.8358887    0.88656896   0.8604832
B-protein    5288  763  660  0.8739051    0.8890383    0.8814068
I-cell_line  1148  244  301  0.82471263   0.79227054   0.80816615
I-RNA        221   60   27   0.78647685   0.891129     0.83553874
B-RNA        157   40   36   0.79695433   0.8134715    0.8051282
B-cell_type  1127  292  250  0.7942213    0.8184459    0.8061516
I-cell_type  1547  392  263  0.7978339    0.85469615   0.82528675
B-DNA        1513  444  387  0.77312213   0.7963158    0.7845475

Macro-average  prec: 0.8069253,  rec: 0.84323275, f1: 0.82467955
Micro-average  prec: 0.82471186, rec: 0.86377037, f1: 0.84378934

```