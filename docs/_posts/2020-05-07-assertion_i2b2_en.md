---
layout: model
title: Detect Assertion Status (DL I2B2)
author: John Snow Labs
name: assertion_i2b2
class: AssertionDLModel
language: en
repository: clinical/models
date: 2020-05-07
tags: [clinical,licensed,assertion,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Assertion of Clinical Entities based on Deep Learning. Identifies the status of predicted entities based on their context.  

## Assertion Status 
``hypothetical``, ``present``, ``absent``, ``possible``, ``conditional``, ``associated_with_someone_else``.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_i2b2_en_2.4.2_2.4_1588811895962.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
clinical_assertion = AssertionDLModel.pretrained("assertion_i2b2", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, nerDLModel, nerConverter, clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([["The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."]]).toDF("text"))

light_model = LightPipeline(model)

```

```scala
...
val clinical_assertion = AssertionDLModel.pretrained("assertion_i2b2", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, nerDLModel, nerConverter, clinical_assertion))

val result = pipeline.fit(Seq.empty["The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.', 'Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population.', 'The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively.', 'We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair', '(bp) insertion/deletion.', 'Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle.', 'The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."].toDS.toDF("text")).transform(data)
```
</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an ``"assertion"`` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select ``"ner_chunk.result"`` and ``"assertion.result"`` from your output dataframe.

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
|----------------|----------------------------------|
| Name:           | assertion_i2b2                   |
| Type:    | AssertionDLModel                 |
| Compatibility:  | Spark NLP 2.4.2+                           |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [document, chunk, word_embeddings] |
|Output labels:        | [assertion]                        |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with ``embeddings_clinical``.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Benchmarking

```bash
                              precision    recall  f1-score

                      absent       0.66      0.72      0.69 
associated_with_someone_else       0.00      0.00      0.00
                 conditional       0.00      0.00      0.00 
                hypothetical       0.92      0.75      0.83 
                    possible       0.00      0.00      0.00 
                     present       0.92      0.84      0.88 

                    accuracy                           0.81 
                   macro avg       0.42      0.39      0.40 
                weighted avg       0.86      0.81      0.83 
                
```