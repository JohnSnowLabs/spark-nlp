---
layout: model
title: Part of Speech for Arabic
author: John Snow Labs
name: pos_ud_padt
date: 2020-11-30
task: Part of Speech Tagging
language: ar
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, ar]
supported: true
annotator: PerceptronModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 15 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_padt_ar_2.7.0_2.4_1606721957579.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_padt", "ar") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate(["كرستيانو رونالدو لاعب برتغالي محترف يلعب في صفوف منتخب البرتغال لكرة القدم"])
```
```scala
...
val pos = PerceptronModel.pretrained("pos_ud_padt", "ar")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("كرستيانو رونالدو لاعب برتغالي محترف يلعب في صفوف منتخب البرتغال لكرة القدم").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""كرستيانو رونالدو لاعب برتغالي محترف يلعب في صفوف منتخب البرتغال لكرة القدم"""]
pos_df = nlu.load('ar.pos').predict(text)
pos_df
```

</div>

## Results

```bash
{'pos': [Annotation(pos, 0, 7, X, {'word': 'كرستيانو'}),
   Annotation(pos, 9, 15, X, {'word': 'رونالدو'}),
   Annotation(pos, 17, 20, NOUN, {'word': 'لاعب'}),
   Annotation(pos, 22, 28, X, {'word': 'برتغالي'}),
   Annotation(pos, 30, 34, X, {'word': 'محترف'}),
   Annotation(pos, 36, 39, VERB, {'word': 'يلعب'}),
   Annotation(pos, 41, 42, ADP, {'word': 'في'}),
   Annotation(pos, 44, 47, NOUN, {'word': 'صفوف'}),
   Annotation(pos, 49, 53, NOUN, {'word': 'منتخب'}),
   Annotation(pos, 55, 62, X, {'word': 'البرتغال'}),
   Annotation(pos, 64, 67, CCONJ, {'word': 'لكرة'}),
   Annotation(pos, 69, 73, NOUN, {'word': 'القدم'})],
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_padt|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[tags, document]|
|Output Labels:|[pos]|
|Language:|ar|

## Data Source

The model is trained on data obtained from [https://universaldependencies.org](https://universaldependencies.org)

## Benchmarking

```bash
|    |              | precision   | recall   |   f1-score |   support |
|---:|:-------------|:------------|:---------|-----------:|----------:|
|  0 | ADJ          | 0.90        | 0.91     |       0.91 |      2937 |
|  1 | ADP          | 0.99        | 1.00     |       0.99 |      4528 |
|  2 | ADV          | 0.96        | 0.93     |       0.95 |       104 |
|  3 | AUX          | 0.88        | 0.85     |       0.87 |       197 |
|  4 | CCONJ        | 1.00        | 0.99     |       0.99 |      1963 |
|  5 | DET          | 0.95        | 0.96     |       0.96 |       623 |
|  6 | NOUN         | 0.94        | 0.96     |       0.95 |      9547 |
|  7 | NUM          | 0.98        | 0.97     |       0.98 |       779 |
|  8 | None         | 1.00        | 1.00     |       1    |      3868 |
|  9 | PART         | 0.92        | 0.93     |       0.93 |       226 |
| 10 | PRON         | 0.99        | 1.00     |       1    |      1133 |
| 11 | PROPN        | 1.00        | 0.48     |       0.65 |        31 |
| 12 | PUNCT        | 1.00        | 1.00     |       1    |      2052 |
| 13 | SCONJ        | 0.99        | 0.98     |       0.98 |       534 |
| 14 | SYM          | 1.00        | 0.98     |       0.99 |        41 |
| 15 | VERB         | 0.94        | 0.93     |       0.94 |      2189 |
| 16 | X            | 0.80        | 0.64     |       0.71 |      1380 |
| 17 | accuracy     |             |          |       0.96 |     32132 |
| 18 | macro avg    | 0.95        | 0.91     |       0.93 |     32132 |
| 19 | weighted avg | 0.95        | 0.96     |       0.95 |     32132 |
```