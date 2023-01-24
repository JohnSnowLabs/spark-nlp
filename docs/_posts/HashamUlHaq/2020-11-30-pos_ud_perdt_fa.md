---
layout: model
title: Part of Speech for Persian
author: John Snow Labs
name: pos_ud_perdt
date: 2020-11-30
task: Part of Speech Tagging
language: fa
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [fa, pos]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_perdt_fa_2.7.0_2.4_1606724821106.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_perdt_fa_2.7.0_2.4_1606724821106.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_perdt", "fa") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate(["جان اسنو جدا از سلطنت شمال ، یک پزشک انگلیسی و رهبر توسعه بیهوشی و بهداشت پزشکی است."])
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_perdt", "fa")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("جان اسنو جدا از سلطنت شمال ، یک پزشک انگلیسی و رهبر توسعه بیهوشی و بهداشت پزشکی است.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""جان اسنو جدا از سلطنت شمال ، یک پزشک انگلیسی و رهبر توسعه بیهوشی و بهداشت پزشکی است"""]
pos_df = nlu.load('fa.pos').predict(text)
pos_df
```

</div>

## Results

```bash
{'pos': [Annotation(pos, 0, 2, NOUN, {'word': 'جان'}),
   Annotation(pos, 4, 7, NOUN, {'word': 'اسنو'}),
   Annotation(pos, 9, 11, ADJ, {'word': 'جدا'}),
   Annotation(pos, 13, 14, ADP, {'word': 'از'}),
   Annotation(pos, 16, 20, NOUN, {'word': 'سلطنت'}),
   Annotation(pos, 22, 25, NOUN, {'word': 'شمال'}),
   Annotation(pos, 27, 27, PUNCT, {'word': '،'}),
   Annotation(pos, 29, 30, NUM, {'word': 'یک'}),
   Annotation(pos, 32, 35, NOUN, {'word': 'پزشک'}),
   Annotation(pos, 37, 43, ADJ, {'word': 'انگلیسی'}),
   Annotation(pos, 45, 45, CCONJ, {'word': 'و'}),
   Annotation(pos, 47, 50, NOUN, {'word': 'رهبر'}),
   Annotation(pos, 52, 56, NOUN, {'word': 'توسعه'}),
   Annotation(pos, 58, 63, VERB, {'word': 'بیهوشی'}),
   Annotation(pos, 65, 65, CCONJ, {'word': 'و'}),
   Annotation(pos, 67, 72, NOUN, {'word': 'بهداشت'}),
   Annotation(pos, 74, 78, ADJ, {'word': 'پزشکی'}),
   Annotation(pos, 80, 82, AUX, {'word': 'است'}),
   Annotation(pos, 83, 83, PUNCT, {'word': '.'})]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_perdt|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[tags, document]|
|Output Labels:|[pos]|
|Language:|fa|

## Data Source

The model is trained on data obtained from [https://universaldependencies.org](https://universaldependencies.org)

## Benchmarking

```bash
|    |              | precision   | recall   |   f1-score |   support |
|---:|:-------------|:------------|:---------|-----------:|----------:|
|  0 | ADJ          | 0.88        | 0.88     |       0.88 |      1647 |
|  1 | ADP          | 0.99        | 0.99     |       0.99 |      3402 |
|  2 | ADV          | 0.94        | 0.91     |       0.92 |       383 |
|  3 | AUX          | 0.99        | 0.99     |       0.99 |      1000 |
|  4 | CCONJ        | 1.00        | 1.00     |       1    |      1022 |
|  5 | DET          | 0.94        | 0.96     |       0.95 |       490 |
|  6 | INTJ         | 0.88        | 0.81     |       0.85 |        27 |
|  7 | NOUN         | 0.95        | 0.96     |       0.95 |      8201 |
|  8 | NUM          | 0.94        | 0.97     |       0.96 |       293 |
|  9 | None         | 1.00        | 0.99     |       0.99 |       289 |
| 10 | PART         | 1.00        | 0.86     |       0.92 |        28 |
| 11 | PRON         | 0.98        | 0.97     |       0.98 |      1117 |
| 12 | PROPN        | 0.84        | 0.78     |       0.81 |      1107 |
| 13 | PUNCT        | 1.00        | 1.00     |       1    |      2134 |
| 14 | SCONJ        | 0.98        | 0.98     |       0.98 |       630 |
| 15 | VERB         | 0.99        | 0.99     |       0.99 |      2581 |
| 16 | accuracy     |             |          |       0.96 |     24351 |
| 17 | macro avg    | 0.96        | 0.94     |       0.95 |     24351 |
| 18 | weighted avg | 0.96        | 0.96     |       0.96 |     24351 |
```