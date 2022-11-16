---
layout: model
title: Detect Drugs and Posology Entities (ner_posology_greedy)
author: John Snow Labs
name: ner_posology_greedy
date: 2020-12-08
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.6.5
spark_version: 2.4
tags: [ner, licensed, clinical, en]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects drugs, dosage, form, frequency, duration, route, and drug strength in text. It differs from `ner_posology` in the sense that it chunks together drugs, dosage, form, strength, and route when they appear together, resulting in a bigger chunk. It is trained using `embeddings_clinical` so please use the same embeddings in the pipeline.

## Predicted Entities

`DRUG`, `STRENGTH`, `DURATION`, `FREQUENCY`, `FORM`, `DOSAGE`, `ROUTE`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_POSOLOGY.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_greedy_en_2.6.4_2.4_1607422064676.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_posology_greedy", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day."]]).toDF("text"))
```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val clinical_ner = NerDLModel.pretrained("ner_posology_greedy", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter))
val data = Seq("The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.posology.greedy").predict("""The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.""")
```

</div>

## Results

```bash
+----+----------------------------------+---------+-------+------------+
|    | chunks                           |   begin |   end | entities   |
|---:|---------------------------------:|--------:|------:|-----------:|
|  0 | 1 capsule of Advil 10 mg         |      27 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO |      67 |    98 | DRUG       |
|  2 | for 5 days                       |      52 |    61 | DURATION   |
|  3 | 40 units of insulin glargine     |     168 |   195 | DRUG       |
|  4 | at night                         |     197 |   204 | FREQUENCY  |
|  5 | 12 units of insulin lispro       |     207 |   232 | DRUG       |
|  6 | with meals                       |     234 |   243 | FREQUENCY  |
|  7 | metformin 1000 mg                |     250 |   266 | DRUG       |
|  8 | two times a day                  |     268 |   282 | FREQUENCY  |
+----+----------------------------------+---------+-------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_greedy|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.5+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on augmented i2b2_med7 + FDA dataset with ``embeddings_clinical``, [https://www.i2b2.org/NLP/Medication](https://www.i2b2.org/NLP/Medication).

## Benchmarking

| label         | tp     | fp    | fn    | prec       | rec        | f1         |
|---------------|--------|-------|-------|------------|------------|------------|
| B-DRUG        | 29362  | 1679  | 1985  | 0.9459103  | 0.93667656 | 0.94127077 |
| B-STRENGTH    | 14018  | 1172  | 864   | 0.922844   | 0.9419433  | 0.9322958  |
| I-DURATION    | 6404   | 935   | 476   | 0.87259847 | 0.93081397 | 0.9007666  |
| I-STRENGTH    | 16686  | 1991  | 1292  | 0.8933983  | 0.9281344  | 0.9104351  |
| I-FREQUENCY   | 19743  | 1088  | 1081  | 0.9477702  | 0.94808877 | 0.9479294  |
| B-FORM        | 2733   | 526   | 780   | 0.8386008  | 0.7779676  | 0.80714715 |
| B-DOSAGE      | 2774   | 474   | 688   | 0.85406405 | 0.80127096 | 0.8268257  |
| I-DOSAGE      | 1357   | 490   | 844   | 0.7347049  | 0.6165379  | 0.67045456 |
| I-DRUG        | 37846  | 4103  | 3386  | 0.90219074 | 0.91787934 | 0.9099674  |
| I-ROUTE       | 208    | 30    | 62    | 0.8739496  | 0.77037036 | 0.8188976  |
| B-ROUTE       | 3061   | 340   | 451   | 0.9000294  | 0.87158316 | 0.88557786 |
| B-DURATION    | 2491   | 388   | 276   | 0.865231   | 0.900253   | 0.8823946  |
| B-FREQUENCY   | 13065  | 608   | 436   | 0.9555328  | 0.9677061  | 0.9615809  |
| I-FORM        | 154    | 69    | 386   | 0.69058293 | 0.2851852  | 0.40366974 |
|               |        |       |       |            |            |            |
| Macro-average | 149902 | 13893 | 13007 | 0.8712434  | 0.82817215 | 0.849162   |
| Micro-average | 149902 | 13893 | 13007 | 0.91518056 | 0.92015785 | 0.9176625  |
