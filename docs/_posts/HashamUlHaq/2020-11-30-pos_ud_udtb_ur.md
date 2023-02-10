---
layout: model
title: Part of Speech for Urdu
author: John Snow Labs
name: pos_ud_udtb
date: 2020-11-30
task: Part of Speech Tagging
language: ur
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, ur]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_udtb_ur_2.7.0_2.4_1606733090479.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_udtb_ur_2.7.0_2.4_1606733090479.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_udtb", "ur") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate(["شمال کا بادشاہ ہونے کے علاوہ ، جان سن ایک انگریزی معالج ہے۔"])

```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_udtb", "ur")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("شمال کا بادشاہ ہونے کے علاوہ ، جان سن ایک انگریزی معالج ہے۔").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""شمال کا بادشاہ ہونے کے علاوہ ، جان سن ایک انگریزی معالج ہے۔"""]
pos_df = nlu.load('ur.pos.ud_udtb').predict(text)
pos_df
```

</div>

## Results

```bash
{'pos': [Annotation(pos, 0, 3, NOUN, {'word': 'شمال'}),
Annotation(pos, 5, 6, ADP, {'word': 'کا'}),
Annotation(pos, 8, 13, NOUN, {'word': 'بادشاہ'}),
Annotation(pos, 15, 18, VERB, {'word': 'ہونے'}),
Annotation(pos, 20, 21, ADP, {'word': 'کے'}),
Annotation(pos, 23, 27, ADP, {'word': 'علاوہ'}),
Annotation(pos, 29, 29, PUNCT, {'word': '،'}),
Annotation(pos, 31, 33, PROPN, {'word': 'جان'}),
Annotation(pos, 35, 36, PROPN, {'word': 'سن'}),
Annotation(pos, 38, 40, NUM, {'word': 'ایک'}),
Annotation(pos, 42, 48, PROPN, {'word': 'انگریزی'}),
Annotation(pos, 50, 54, ADJ, {'word': 'معالج'}),
Annotation(pos, 56, 58, PUNCT, {'word': 'ہے۔'})]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_udtb|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[tags, document]|
|Output Labels:|[pos]|
|Language:|ur|

## Data Source

The model is trained on data obtained from [https://universaldependencies.org](https://universaldependencies.org)

## Benchmarking

```bash
|    |              | precision   | recall   |   f1-score |   support |
|---:|:-------------|:------------|:---------|-----------:|----------:|
|  0 | ADJ          | 0.84        | 0.85     |       0.84 |      1117 |
|  1 | ADP          | 0.98        | 0.99     |       0.98 |      3122 |
|  2 | ADV          | 0.83        | 0.65     |       0.73 |       125 |
|  3 | AUX          | 0.97        | 0.96     |       0.96 |       937 |
|  4 | CCONJ        | 0.96        | 1.00     |       0.98 |       338 |
|  5 | DET          | 0.87        | 0.82     |       0.84 |       237 |
|  6 | NOUN         | 0.89        | 0.92     |       0.9  |      3690 |
|  7 | NUM          | 0.97        | 0.95     |       0.96 |       267 |
|  8 | PART         | 0.96        | 0.88     |       0.91 |       337 |
|  9 | PRON         | 0.96        | 0.94     |       0.95 |       499 |
| 10 | PROPN        | 0.88        | 0.85     |       0.86 |      1975 |
| 11 | PUNCT        | 1.00        | 1.00     |       1    |       682 |
| 12 | SCONJ        | 0.97        | 0.99     |       0.98 |       248 |
| 13 | VERB         | 0.95        | 0.95     |       0.95 |      1232 |
| 14 | accuracy     |             |          |       0.93 |     14806 |
| 15 | macro avg    | 0.93        | 0.91     |       0.92 |     14806 |
| 16 | weighted avg | 0.93        | 0.93     |       0.93 |     14806 |
​
```