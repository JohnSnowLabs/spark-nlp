---
layout: model
title: Relation Extraction Model Clinical
author: John Snow Labs
name: re_clinical
class: RelationExtractionModel
language: en
repository: clinical/models
date: 2020-09-24
tags: [clinical,relation extraction,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Relation Extraction model based on syntactic features using deep learning. Models the set of clinical relations defined in the 2010 ``i2b2`` relation challenge.

## Predicted Entities 
*TrIP*: A certain treatment has improved or cured a medical problem (eg, ‘infection resolved with antibiotic course’)
*TrWP*: A patient's medical problem has deteriorated or worsened because of or in spite of a treatment being administered (eg, ‘the tumor was growing despite the drain’)
*TrCP*: A treatment caused a medical problem (eg, ‘penicillin causes a rash’)
*TrAP*: A treatment administered for a medical problem (eg, ‘Dexamphetamine for narcolepsy’)
*TrNAP*: The administration of a treatment was avoided because of a medical problem (eg, ‘Ralafen which is contra-indicated because of ulcers’)
*TeRP*: A test has revealed some medical problem (eg, ‘an echocardiogram revealed a pericardial effusion’)
*TeCP*: A test was performed to investigate a medical problem (eg, ‘chest x-ray done to rule out pneumonia’)
*PIP*: Two problems are related to each other (eg, ‘Azotemia presumed secondary to sepsis’)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1600987935304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

reModel = RelationExtractionModel.pretrained("re_clinical","en","clinical/models")\
    .setInputCols(["word_embeddings","chunk","pos","dependency"])\
    .setOutput("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel])
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = LightPipeline(model).fullAnnotate("""The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily.He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.""")
```

```scala
...

val model = RelationExtractionModel.pretrained("re_clinical","en","clinical/models")
    .setInputCols("word_embeddings","chunk","pos","dependency")
    .setOutputCol("relations")
    
val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel))

val result = pipeline.fit(Seq.empty["""The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily.He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals , and metformin 1000 mg two times a day."""].toDS.toDF("text")).transform(data)

```
</div>

{:.h2_title}
## Results
```bash
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
|   |       relation | entity1 | entity1_begin | entity1_end |           chunk1 |   entity2 | entity2_begin | entity2_end |           chunk2 | confidence |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 0 |    DOSAGE-DRUG |  DOSAGE |            28 |          33 |           1 unit |      DRUG |            38 |          42 |            Advil |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 1 |  DRUG-DURATION |    DRUG |            38 |          42 |            Advil |  DURATION |            44 |          53 |       for 5 days |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 2 |    DOSAGE-DRUG |  DOSAGE |            96 |         101 |           1 unit |      DRUG |           106 |         114 |        Metformin |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 3 | DRUG-FREQUENCY |    DRUG |           106 |         114 |        Metformin | FREQUENCY |           116 |         120 |            daily |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 4 |    DOSAGE-DRUG |  DOSAGE |           190 |         197 |         40 units |      DRUG |           202 |         217 | insulin glargine |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 5 | DRUG-FREQUENCY |    DRUG |           202 |         217 | insulin glargine | FREQUENCY |           219 |         226 |         at night |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 6 |    DOSAGE-DRUG |  DOSAGE |           231 |         238 |         12 units |      DRUG |           243 |         256 |   insulin lispro |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 7 | DRUG-FREQUENCY |    DRUG |           243 |         256 |   insulin lispro | FREQUENCY |           258 |         267 |       with meals |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 8 |  DRUG-STRENGTH |    DRUG |           275 |         283 |        metformin |  STRENGTH |           285 |         291 |          1000 mg |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
| 9 | DRUG-FREQUENCY |    DRUG |           275 |         283 |        metformin | FREQUENCY |           293 |         307 |  two times a day |        1.0 |
+---+----------------+---------+---------------+-------------+------------------+-----------+---------------+-------------+------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------|
| Name:           | re_clinical                             |
| Type:    | RelationExtractionModel                 |
| Compatibility:  | Spark NLP 2.5.5+                                   |
| License:        | Licensed                                |
|Edition:|Official|                              |
|Input labels:         | [word_embeddings, chunk, pos, dependency] |
|Output labels:        | [category]                                |
| Language:       | en                                      |
| Case sensitive: | False                                   |
| Dependencies:  | embeddings_clinical                     |

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking
The model has been validated agains the posology dataset described in (Magge, Scotch, & Gonzalez-Hernandez, 2018).
```bash
+----------------+--------+-----------+------+------------------------------------------------+
|    Relation    | Recall | Precision |  F1  | F1 (Magge, Scotch, & Gonzalez-Hernandez, 2018) |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-ADE       | 0.66   | 1.00      | 0.80 | 0.76                                           |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-DOSAGE    | 0.89   | 1.00      | 0.94 | 0.91                                           |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-DURATION  | 0.75   | 1.00      | 0.85 | 0.92                                           |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-FORM      | 0.88   | 1.00      | 0.94 | 0.95*                                          |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-FREQUENCY | 0.79   | 1.00      | 0.88 | 0.90                                           |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-REASON    | 0.60   | 1.00      | 0.75 | 0.70                                           |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-ROUTE     | 0.79   | 1.00      | 0.88 | 0.95*                                          |
+----------------+--------+-----------+------+------------------------------------------------+
| DRUG-STRENGTH  | 0.95   | 1.00      | 0.98 | 0.97                                           |
+----------------+--------+-----------+------+------------------------------------------------+
```
*Magge, Scotch, Gonzalez-Hernandez (2018) collapsed DRUG-FORM and DRUG-ROUTE into a single relation.