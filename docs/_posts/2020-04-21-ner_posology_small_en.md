---
layout: model
title: Detect Drug Information (Small).
author: John Snow Labs
name: ner_posology_small
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,licensed,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---
 
{:.h2_title}
## Description

Pretrained named entity recognition deep learning model for posology, this NER is trained with the ``embeddings_clinical`` word embeddings model, so be sure to use the same embeddings in the pipeline.

{:.h2_title}
## Predicted Entities 
``DOSAGE``, ``DRUG``, ``DURATION``, ``FORM``, ``FREQUENCY``, ``ROUTE``, ``STRENGTH``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_small_en_2.4.2_2.4_1587513301751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

clinical_ner = NerDLModel.pretrained("ner_posology_small","en","clinical/models")\
	.setInputCols(["sentence","token","word_embeddings"])\
	.setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, licensed,clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."""]})))

```

```scala
...

val model = NerDLModel.pretrained("ner_posology_small","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter))

val result = pipeline.fit(Seq.empty["""The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."""].toDS.toDF("text")).transform(data)

```
</div>

## Results
```bash
+--------------+---------+
|chunk         |ner      |
+--------------+---------+
|insulin       |DRUG     |
|Bactrim       |DRUG     |
|for 14 days   |DURATION |
|Fragmin       |DRUG     |
|5000 units    |DOSAGE   |
|subcutaneously|ROUTE    |
|daily         |FREQUENCY|
|Xenaderm      |DRUG     |
|topically     |ROUTE    |
|b.i.d.,       |FREQUENCY|
|Lantus        |DRUG     |
|40 units      |DOSAGE   |
|subcutaneously|ROUTE    |
|at bedtime    |FREQUENCY|
|OxyContin     |DRUG     |
|30 mg         |STRENGTH |
|p.o           |ROUTE    |
|q.12 h        |FREQUENCY|
|folic acid    |DRUG     |
|1 mg          |STRENGTH |
+--------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_posology_small               |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.4.2+                           |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | sentence, token, word_embeddings |
|Output labels:        | ner                              |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset (no FDA) with ``embeddings_clinical``.
https://www.i2b2.org/NLP/Medication

{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-DRUG        |  1408 |    62 |    99 | 0.957823 | 0.934307 | 0.945919 |
|  1 | B-STRENGTH    |   470 |    43 |    29 | 0.916179 | 0.941884 | 0.928854 |
|  2 | I-DURATION    |   123 |    22 |     8 | 0.848276 | 0.938931 | 0.891304 |
|  3 | I-STRENGTH    |   499 |    66 |    15 | 0.883186 | 0.970817 | 0.924931 |
|  4 | I-FREQUENCY   |   945 |    47 |    55 | 0.952621 | 0.945    | 0.948795 |
|  5 | B-FORM        |   365 |    13 |    12 | 0.965608 | 0.96817  | 0.966887 |
|  6 | B-DOSAGE      |   298 |    27 |    26 | 0.916923 | 0.919753 | 0.918336 |
|  7 | I-DOSAGE      |   348 |    29 |    22 | 0.923077 | 0.940541 | 0.931727 |
|  8 | I-DRUG        |   208 |    25 |    60 | 0.892704 | 0.776119 | 0.830339 |
|  9 | I-ROUTE       |    10 |     0 |     2 | 1        | 0.833333 | 0.909091 |
| 10 | B-ROUTE       |   467 |     4 |    25 | 0.991507 | 0.949187 | 0.969886 |
| 11 | B-DURATION    |    64 |    10 |    10 | 0.864865 | 0.864865 | 0.864865 |
| 12 | B-FREQUENCY   |   588 |    12 |    17 | 0.98     | 0.971901 | 0.975934 |
| 13 | I-FORM        |   264 |     5 |     4 | 0.981413 | 0.985075 | 0.98324  |
| 14 | Macro-average |  6057 |   365 |   384 | 0.93387  | 0.924277 | 0.929049 |
| 15 | Micro-average |  6057 |   365 |   384 | 0.943164 | 0.940382 | 0.941771 |
```