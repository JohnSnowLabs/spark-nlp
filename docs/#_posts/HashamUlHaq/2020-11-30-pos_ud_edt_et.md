---
layout: model
title: Part of Speech for Estonian
author: John Snow Labs
name: pos_ud_edt
date: 2020-11-30
task: Part of Speech Tagging
language: et
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [et, pos]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_edt_et_2.7.0_2.4_1606724297129.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_edt", "et") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate(["Lisaks sellele, et ta on põhjamaa kuningas, on John Snow inglise arst ning narkoosi ja meditsiinilise hügieeni arendamise juht."])

```
```scala
...
val pos = PerceptronModel.pretrained("pos_ud_edt", "et")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Lisaks sellele, et ta on põhjamaa kuningas, on John Snow inglise arst ning narkoosi ja meditsiinilise hügieeni arendamise juht.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
{'pos': [Annotation(pos, 0, 5, NOUN, {'word': 'Lisaks'}),
   Annotation(pos, 7, 13, PRON, {'word': 'sellele'}),
   Annotation(pos, 14, 14, PUNCT, {'word': ','}),
   Annotation(pos, 16, 17, SCONJ, {'word': 'et'}),
   Annotation(pos, 19, 20, PRON, {'word': 'ta'}),
   Annotation(pos, 22, 23, AUX, {'word': 'on'}),
   Annotation(pos, 25, 32, NOUN, {'word': 'põhjamaa'}),
   Annotation(pos, 34, 41, NOUN, {'word': 'kuningas'}),
   Annotation(pos, 42, 42, PUNCT, {'word': ','}),
   Annotation(pos, 44, 45, AUX, {'word': 'on'}),
   Annotation(pos, 47, 50, PROPN, {'word': 'John'}),
   Annotation(pos, 52, 55, PROPN, {'word': 'Snow'}),
   Annotation(pos, 57, 63, ADJ, {'word': 'inglise'}),
   Annotation(pos, 65, 68, NOUN, {'word': 'arst'}),
   Annotation(pos, 70, 73, CCONJ, {'word': 'ning'}),
   Annotation(pos, 75, 82, NOUN, {'word': 'narkoosi'}),
   Annotation(pos, 84, 85, CCONJ, {'word': 'ja'}),
   Annotation(pos, 87, 100, NOUN, {'word': 'meditsiinilise'}),
   Annotation(pos, 102, 109, NOUN, {'word': 'hügieeni'}),
   Annotation(pos, 111, 120, NOUN, {'word': 'arendamise'}),
   Annotation(pos, 122, 125, NOUN, {'word': 'juht'}),
   Annotation(pos, 126, 126, PUNCT, {'word': '.'})]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_edt|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[tags, document]|
|Output Labels:|[pos]|
|Language:|et|

## Data Source

The model is trained on data obtained from [https://universaldependencies.org](https://universaldependencies.org)

## Benchmarking

```bash
|    |              | precision   | recall   |   f1-score |   support |
|---:|:-------------|:------------|:---------|-----------:|----------:|
|  0 | ADJ          | 0.86        | 0.82     |       0.84 |      3655 |
|  1 | ADP          | 0.91        | 0.92     |       0.91 |       838 |
|  2 | ADV          | 0.95        | 0.95     |       0.95 |      4553 |
|  3 | AUX          | 0.94        | 0.98     |       0.96 |      2426 |
|  4 | CCONJ        | 0.99        | 0.98     |       0.98 |      1820 |
|  5 | DET          | 0.82        | 0.74     |       0.78 |       752 |
|  6 | INTJ         | 0.92        | 0.68     |       0.78 |        50 |
|  7 | NOUN         | 0.92        | 0.95     |       0.94 |     11352 |
|  8 | NUM          | 0.96        | 0.90     |       0.93 |       756 |
|  9 | PRON         | 0.93        | 0.94     |       0.93 |      2350 |
| 10 | PROPN        | 0.96        | 0.92     |       0.94 |      2619 |
| 11 | PUNCT        | 1.00        | 1.00     |       1    |      6989 |
| 12 | SCONJ        | 0.96        | 0.99     |       0.98 |      1048 |
| 13 | SYM          | 1.00        | 0.72     |       0.84 |        18 |
| 14 | VERB         | 0.93        | 0.91     |       0.92 |      4846 |
| 15 | X            | 0.56        | 0.15     |       0.23 |        68 |
| 16 | accuracy     |             |          |       0.94 |     44140 |
| 17 | macro avg    | 0.91        | 0.85     |       0.87 |     44140 |
| 18 | weighted avg | 0.94        | 0.94     |       0.94 |     44140 |
```