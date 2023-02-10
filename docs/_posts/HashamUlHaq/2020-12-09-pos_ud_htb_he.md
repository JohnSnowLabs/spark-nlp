---
layout: model
title: Part of Speech for Hebrew
author: John Snow Labs
name: pos_ud_htb
date: 2020-12-09
task: Part of Speech Tagging
language: he
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, open_source, he]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_htb_he_2.7.0_2.4_1607521333296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_htb_he_2.7.0_2.4_1607521333296.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of an nlp pipeline after tokenization.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_htb", "he") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate(["ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , שגויס לארגון הפת"ח והופעל על ידי חיזבאללה"])

```
```scala
...
val pos = PerceptronModel.pretrained("pos_ud_htb", "he")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , שגויס לארגון הפת"ח והופעל על ידי חיזבאללה").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , שגויס לארגון הפת"ח והופעל על ידי חיזבאללה"]
pos_df = nlu.load('he.pos.ud_htb').predict(text, output_level='token')
pos_df
```

</div>

## Results

```bash
{'pos': [Annotation(pos, 0, 0, ADP, {'word': 'ב'}),
Annotation(pos, 1, 1, PUNCT, {'word': '-'}),
Annotation(pos, 3, 4, NUM, {'word': '25'}),
Annotation(pos, 6, 12, VERB, {'word': 'לאוגוסט'}),
Annotation(pos, 14, 16, None, {'word': 'עצר'}),
Annotation(pos, 18, 22, VERB, {'word': 'השב"כ'}),
Annotation(pos, 24, 25, ADP, {'word': 'את'}),
Annotation(pos, 27, 31, PROPN, {'word': 'מוחמד'}),
Annotation(pos, 33, 42, PROPN, {'word': "אבו-ג'וייד"}),
Annotation(pos, 44, 44, PUNCT, {'word': ','}),
Annotation(pos, 46, 49, NOUN, {'word': 'אזרח'}),
Annotation(pos, 51, 55, ADJ, {'word': 'ירדני'}),
Annotation(pos, 57, 57, PUNCT, {'word': ','}),
Annotation(pos, 59, 63, VERB, {'word': 'שגויס'}),
Annotation(pos, 65, 70, ADP, {'word': 'לארגון'}),
Annotation(pos, 72, 76, NOUN, {'word': 'הפת"ח'}),
Annotation(pos, 78, 83, PROPN, {'word': 'והופעל'}),
Annotation(pos, 85, 86, ADP, {'word': 'על'}),
Annotation(pos, 88, 90, NOUN, {'word': 'ידי'}),
Annotation(pos, 92, 99, PROPN, {'word': 'חיזבאללה'})]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_htb|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[tags, document]|
|Output Labels:|[pos]|
|Language:|he|

## Data Source

The model is trained on data obtained from [https://universaldependencies.org](https://universaldependencies.org)

## Benchmarking

```bash
|    |              | precision   | recall   |   f1-score |   support |
|---:|:-------------|:------------|:---------|-----------:|----------:|
|  0 | ADJ          | 0.83        | 0.83     |       0.83 |       676 |
|  1 | ADP          | 0.99        | 0.99     |       0.99 |      1889 |
|  2 | ADV          | 0.93        | 0.89     |       0.91 |       408 |
|  3 | AUX          | 0.90        | 0.90     |       0.9  |       229 |
|  4 | CCONJ        | 0.97        | 0.99     |       0.98 |       434 |
|  5 | DET          | 0.97        | 0.99     |       0.98 |      1390 |
|  6 | NOUN         | 0.91        | 0.94     |       0.93 |      3056 |
|  7 | NUM          | 0.97        | 0.96     |       0.97 |       285 |
|  9 | PRON         | 0.97        | 0.99     |       0.98 |       443 |
| 10 | PROPN        | 0.82        | 0.72     |       0.77 |       573 |
| 11 | PUNCT        | 1.00        | 1.00     |       1    |      1381 |
| 12 | SCONJ        | 0.99        | 0.90     |       0.94 |       411 |
| 13 | VERB         | 0.87        | 0.85     |       0.86 |      1063 |
| 14 | X            | 1.00        | 0.17     |       0.29 |         6 |
| 15 | accuracy     |             |          |       0.95 |     15089 |
| 16 | macro avg    | 0.94        | 0.87     |       0.89 |     15089 |
| 17 | weighted avg | 0.95        | 0.95     |       0.95 |     15089 |

```