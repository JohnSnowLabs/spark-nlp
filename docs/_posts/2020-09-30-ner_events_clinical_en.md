---
layout: model
title: Detect Clinical Events
author: John Snow Labs
name: ner_events_clinical
date: 2020-09-30
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.5.5
tags: [ner, en, licensed, clinical]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to detect clinical events in medical text.

## Predicted Entities
`DATE`, `TIME`, `PROBLEM`, `TEST`, `TREATMENT`, `OCCURENCE`, `CLINICAL_DEPT`, `EVIDENTIAL`, `DURATION`, `FREQUENCY`, `ADMISSION`, `DISCHARGE`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_en_2.5.5_2.4_1597775531760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_events_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("The patient presented to the emergency room last evening")

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_events_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings") 
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val result = pipeline.fit(Seq.empty["The patient presented to the emergency room last evening"].toDS.toDF("text")).transform(data)

```

</div>

{:.h2_title}
## Results

```bash
+----+-----------------------------+---------+---------+-----------------+
|    | chunk                       |   begin |   end   |     entity      |
+====+=============================+=========+=========+=================+
|  0 | presented                   |    12   |    20   |   EVIDENTIAL    |
+----+-----------------------------+---------+---------+-----------------+
|  1 | the emergency room          |    25   |    42   |  CLINICAL_DEPT  |
+----+-----------------------------+---------+---------+-----------------+
|  2 | last evening                |    44   |    55   |     DATE        |
+----+-----------------------------+---------+---------+-----------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_events_clinical|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.5.5 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on i2b2 events data with *clinical_embeddings*.

## Benchmarking
```bash
|    | label           |     tp |    fp |   fn |     prec |      rec |        f1 |
|---:|----------------:|-------:|------:|-----:|---------:|---------:|----------:|
|  0 | I-TIME          |     82 |    12 |   45 | 0.87234  | 0.645669 | 0.742081  |
|  1 | I-EVIDENTIAL    |      0 |     3 |   18 | 0        | 0        | 0         |
|  2 | I-TREATMENT     |   2580 |   439 |  535 | 0.854588 | 0.82825  | 0.841213  |
|  3 | B-OCCURRENCE    |   1548 |   680 |  945 | 0.694793 | 0.620939 | 0.655793  |
|  4 | I-DURATION      |    366 |   183 |   99 | 0.666667 | 0.787097 | 0.721893  |
|  5 | B-DATE          |    847 |   151 |  138 | 0.848697 | 0.859898 | 0.854261  |
|  6 | I-DATE          |    921 |   191 |  196 | 0.828237 | 0.82453  | 0.82638   |
|  7 | B-ADMISSION     |    105 |   102 |   15 | 0.507246 | 0.875    | 0.642202  |
|  8 | I-PROBLEM       |   5238 |   902 |  823 | 0.853094 | 0.864214 | 0.858618  |
|  9 | B-CLINICAL_DEPT |    613 |   130 |  119 | 0.825034 | 0.837432 | 0.831187  |
| 10 | B-TIME          |     36 |     8 |   24 | 0.818182 | 0.6      | 0.692308  |
| 11 | I-CLINICAL_DEPT |   1273 |   210 |  137 | 0.858395 | 0.902837 | 0.880055  |
| 12 | B-PROBLEM       |   3717 |   608 |  591 | 0.859422 | 0.862813 | 0.861114  |
| 13 | I-FREQUENCY     |     64 |    32 |   97 | 0.666667 | 0.397516 | 0.498054  |
| 14 | I-OCCURRENCE    |    726 |   728 |  886 | 0.499312 | 0.450372 | 0.473581  |
| 15 | I-TEST          |   2304 |   384 |  361 | 0.857143 | 0.86454  | 0.860826  |
| 16 | B-TEST          |   1870 |   372 |  300 | 0.834077 | 0.861751 | 0.847688  |
| 17 | B-TREATMENT     |   2767 |   437 |  513 | 0.863608 | 0.843598 | 0.853485  |
| 18 | B-DISCHARGE     |      2 |     1 |  115 | 0.666667 | 0.017094 | 0.0333333 |
| 19 | B-EVIDENTIAL    |    394 |   109 |  201 | 0.7833   | 0.662185 | 0.717669  |
| 20 | B-DURATION      |    236 |   119 |  105 | 0.664789 | 0.692082 | 0.678161  |
| 21 | B-FREQUENCY     |    117 |    20 |   79 | 0.854015 | 0.596939 | 0.702703  |
| 22 | Macro-average   | 25806  | 5821  | 6342 | 0.735285 | 0.677034 | 0.704959  |
| 23 | Micro-average   | 25806  | 5821  | 6342 | 0.815948 | 0.802725 | 0.809283  |
```