---
layout: model
title: Detect 10 Different Entities in Hebrew (hebrew_cc_300d embeddings)
author: John Snow Labs
name: hebrewner_cc_300d
date: 2022-08-09
tags: [he, open_source]
task: Named Entity Recognition
language: he
edition: Spark NLP 4.0.2
spark_version: 3.0
supported: true
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses Hebrew word embeddings to find 10 different types of entities in Hebrew text. It is trained using `hebrew_cc_300d` word embeddings, please use the same embeddings in the pipeline.

Predicted entities: Persons-`PERS`, Dates-`DATE`, Organizations-`ORG`, Locations-`LOC`, Percentage-`PERCENT`, Money-`MONEY`, Time-`TIME`, Miscellaneous (Affiliation)-`MISC_AFF`, Miscellaneous (Event)-`MISC_EVENT`, Miscellaneous (Entity)-`MISC_ENT`.

## Predicted Entities

`PERS`, `LOC`, `DATE`, `ORG`, `TIME`, `MONEY`, `MISC_EVENT`, `MISC_AFF`, `PERCENT`, `MISC_ENT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hebrewner_cc_300d_he_4.0.2_3.0_1660031325511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hebrewner_cc_300d_he_4.0.2_3.0_1660031325511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("hebrew_cc_300d", "he") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner = NerDLModel.pretrained("hebrewner_cc_300d", "he") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, 
                                sentence_detector, 
                                tokenizer, 
                                word_embeddings, 
                                ner, 
                                ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , שגויס לארגון הפת"ח והופעל על ידי חיזבאללה. אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל , לבצע פיגוע ברכבת ישראל בנהריה , לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים.""")
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = new SentenceDetector()
		.setInputCols(Array("document"))
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols(Array("sentence"))
		.setOutputCol("token")
	
val embeddings = WordEmbeddingsModel.pretrained("hebrew_cc_300d", "he",)
		.setInputCols(Array("sentence", "token"))
	  .setOutputCol("embeddings")

val ner = NerDLModel.pretrained("hebrewner_cc_300d", "he")
		.setInputCols(Array("sentence", "token", "embeddings"))
		.setOutputCol("ner")

val ner_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "ner"))
		.setOutputCol("ner_chunk")

val nlp_pipeline  = new Pipeline().setStages(Array(
					documentAssembler, 
					sentenceDetector, 
					tokenizer, 
					embeddings, 
					ner, 
					ner_converter))

val data = Seq("""ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , שגויס לארגון הפת"ח והופעל על ידי חיזבאללה. אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל , לבצע פיגוע ברכבת ישראל בנהריה , לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים.""").toDS.toDF("text")

val result = nlp_pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("he.ner").predict("""ח והופעל על ידי חיזבאללה. אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל , לבצע פיגוע ברכבת ישראל בנהריה , לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים.""")
```
</div>

## Results

```bash
|    | ner_chunk         | entity_label   |
|---:|------------------:|---------------:|
|  0 | 25 לאוגוסט          | DATE           |
|  1 | השב"כ             | ORG            |
|  2 | מוחמד אבו-ג'וייד      | PERS           |
|  3 | ירדני               | MISC_AFF       |
|  4 | הפת"ח             | ORG            |
|  5 | חיזבאללה            | ORG            |
|  6 | אבו-ג'וייד           | PERS           |
|  7 | בגדה               | LOC            |
|  8 | ערביי              | MISC_AFF       |
|  9 | ישראל              | LOC            |
| 10 | ברכבת ישראל         | ORG            |
| 11 | בנהריה              | LOC            |
| 12 | ישראליות            | MISC_AFF       |
| 13 | בירדן              < | LOC            |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hebrewner_cc_300d|
|Type:|ner|
|Compatibility:|Spark NLP 4.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|he|
|Size:|14.8 MB|

## References

This model is trained on dataset obtained from [https://www.cs.bgu.ac.il/~elhadad/nlpproj/naama/](https://www.cs.bgu.ac.il/~elhadad/nlpproj/naama/)

## Benchmarking

```bash
label            tp   fp    fn    prec       rec        f1
I-TIME            5    2     0  0.714286  1         0.833333
I-MISC_AFF        2    0     3  1         0.4       0.571429
B-MISC_EVENT      7    0     1  1         0.875     0.933333
B-LOC           180   24    37  0.882353  0.829493  0.855107
I-ORG           124   47    38  0.725146  0.765432  0.744745
B-DATE           50    4     7  0.925926  0.877193  0.900901
I-PERS          157   10    15  0.94012   0.912791  0.926254
I-DATE           39    7     8  0.847826  0.829787  0.83871 
B-MISC_AFF      132   11     9  0.923077  0.93617   0.929577
I-MISC_EVENT      6    0     2  1         0.75      0.857143
B-TIME            4    0     1  1         0.8       0.888889
I-PERCENT         8    0     0  1         1         1       
I-MISC_ENT       11    3    10  0.785714  0.52381   0.628571
B-MISC_ENT        8    1     5  0.888889  0.615385  0.727273
I-LOC            79   18    23  0.814433  0.77451   0.79397 
B-PERS          231   22    26  0.913044  0.898833  0.905882
B-MONEY          36    2     2  0.947368  0.947368  0.947368
B-PERCENT        28    3     0  0.903226  1         0.949152
B-ORG           166   41    37  0.801932  0.817734  0.809756
I-MONEY          61    1     1  0.983871  0.983871  0.983871
Macro-average  1334  196   225  0.899861  0.826869  0.861822
Micro-average  1334  196   225  0.871895  0.855677  0.86371 
```
