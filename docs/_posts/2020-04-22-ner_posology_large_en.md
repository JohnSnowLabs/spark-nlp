---
layout: model
title: Detect Drug Information (Large)
author: John Snow Labs
name: ner_posology_large_en
date: 2020-04-22
tags: [ner, en, clinical, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for posology. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

## Predicted Entities 
`DOSAGE`, `DRUG`, `DURATION`, `FORM`, `FREQUENCY`, `ROUTE`, `STRENGTH`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_en_2.4.2_2.4_1587513302751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
clinical_ner = NerDLModel.pretrained("ner_posology_large", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, licensed,clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."""
]})))

```

```scala
...

val ner = NerDLModel.pretrained("ner_posology_large", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, licensed,ner, ner_converter))

val result = pipeline.fit(Seq.empty["""The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."""].toDS.toDF("text")).transform(data)

```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.

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
|b.i.d         |FREQUENCY|
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
only showing top 20 rows
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_large_en|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset and FDA Drug datasets with 'embeddings_clinical'.
https://open.fda.gov/

{:.h2_title}
## Benchmarking
```bash
|    | label         |     tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|-------:|------:|------:|---------:|---------:|---------:|
|  0 | B-DRUG        |  30096 |  1952 |  1265 | 0.939091 | 0.959663 | 0.949266 |
|  1 | B-STRENGTH    |  18379 |  1286 |  1142 | 0.934605 | 0.941499 | 0.938039 |
|  2 | I-DURATION    |   6346 |   692 |   391 | 0.901677 | 0.941962 | 0.921379 |
|  3 | I-STRENGTH    |  21368 |  2752 |  1411 | 0.885904 | 0.938057 | 0.911235 |
|  4 | I-FREQUENCY   |  18406 |  1525 |  1112 | 0.923486 | 0.943027 | 0.933154 |
|  5 | B-FORM        |  11297 |  1276 |   726 | 0.898513 | 0.939616 | 0.918605 |
|  6 | B-DOSAGE      |   3731 |   611 |   765 | 0.859281 | 0.829849 | 0.844309 |
|  7 | I-DOSAGE      |   2100 |   734 |   887 | 0.741002 | 0.703047 | 0.721526 |
|  8 | I-DRUG        |  11853 |  1364 |  1202 | 0.8968   | 0.907928 | 0.902329 |
|  9 | I-ROUTE       |    227 |    31 |    56 | 0.879845 | 0.80212  | 0.839187 |
| 10 | B-ROUTE       |   5870 |   436 |   488 | 0.930859 | 0.923246 | 0.927037 |
| 11 | B-DURATION    |   2493 |   313 |   205 | 0.888453 | 0.924018 | 0.905887 |
| 12 | B-FREQUENCY   |  12648 |   709 |   430 | 0.946919 | 0.96712  | 0.956913 |
| 13 | I-FORM        |    919 |   472 |   502 | 0.660676 | 0.646728 | 0.653627 |
| 14 | Macro-average | 145733 | 14153 | 10582 | 0.877651 | 0.88342  | 0.880526 |
| 15 | Micro-average | 145733 | 14153 | 10582 | 0.911481 | 0.932303 | 0.921774 |
```