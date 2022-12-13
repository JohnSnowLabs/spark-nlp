---
layout: model
title: Detect Drug Information
author: John Snow Labs
name: ner_posology_en
date: 2020-04-15
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.4.2
spark_version: 2.4
tags: [ner, en, licensed, clinical]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Pretrained named entity recognition deep learning model for posology. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

{:.h2_title}
## Predicted Entities 
``DOSAGE``, ``DRUG``, ``DURATION``, ``FORM``, ``FREQUENCY``, ``ROUTE``, ``STRENGTH``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_en_2.4.4_2.4_1584452534235.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_en_2.4.4_2.4_1584452534235.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_posology", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d.']], ["text"]))

```

```scala
...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_posology", "en", "clinical/models") 
  .setInputCols(Array("sentence", "token", "embeddings")) 
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d.").toDF("text")
val result = pipeline.fit(data).transform(data)
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
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_en|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset and FDA Drug datasets with ``embeddings_clinical``.
https://open.fda.gov/


## Benchmarking
```bash
|    | label         |     tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:--------------|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | B-DRUG        |   2639 |   221 |  117 | 0.922727 | 0.957547 | 0.939815 |
|  1 | B-STRENGTH    |   1711 |   188 |   87 | 0.901    | 0.951613 | 0.925615 |
|  2 | I-DURATION    |    553 |    74 |   60 | 0.881978 | 0.902121 | 0.891935 |
|  3 | I-STRENGTH    |   1927 |   239 |  176 | 0.889658 | 0.91631  | 0.902788 |
|  4 | I-FREQUENCY   |   1749 |   163 |  133 | 0.914749 | 0.92933  | 0.921982 |
|  5 | B-FORM        |   1028 |   109 |   84 | 0.904134 | 0.92446  | 0.914184 |
|  6 | B-DOSAGE      |    323 |    71 |   81 | 0.819797 | 0.799505 | 0.809524 |
|  7 | I-DOSAGE      |    173 |    89 |   82 | 0.660305 | 0.678431 | 0.669246 |
|  8 | I-DRUG        |   1020 |   129 |  118 | 0.887728 | 0.896309 | 0.891998 |
|  9 | I-ROUTE       |     17 |     4 |    5 | 0.809524 | 0.772727 | 0.790698 |
| 10 | B-ROUTE       |    526 |    49 |   52 | 0.914783 | 0.910035 | 0.912402 |
| 11 | B-DURATION    |    223 |    35 |   27 | 0.864341 | 0.892    | 0.877953 |
| 12 | B-FREQUENCY   |   1170 |    90 |   54 | 0.928571 | 0.955882 | 0.942029 |
| 13 | I-FORM        |     48 |     6 |   11 | 0.888889 | 0.813559 | 0.849558 |
| 14 | Macro-average | 13107  | 1467  | 1087 | 0.870585 | 0.878559 | 0.874554 |
| 15 | Micro-average | 13107  | 1467  | 1087 | 0.899341 | 0.923418 | 0.911221 |
```