---
layout: model
title: Detect Movie Entities - MIT Movie Simple (ner_mit_movie_simple_distilbert_base_cased)
author: John Snow Labs
name: ner_mit_movie_simple_distilbert_base_cased
date: 2021-07-20
tags: [open_source, en, english, distibert, movie, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This NER model was trained over the MIT Movie Corpus simple queries dataset to detect movie trivia. We used DistilBertEmbeddings (distilbert_base_cased) model for the embeddings to train this NER model.

## Predicted Entities

- ACTOR
- CHARACTER
- DIRECTOR
- GENRE
- PLOT
- RATING
- RATINGS_AVERAGE
- REVIEW
- SONG
- TITLE
- TRAILER
- YEAR

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_mit_movie_simple_distilbert_base_cased_en_3.1.3_2.4_1626778585112.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

embeddings = DistilBertEmbeddings\
.pretrained('distilbert_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_mit_movie_simple_distilbert_base_cased', 'en') \
.setInputCols(['document', 'token', 'embeddings']) \
.setOutputCol('ner')

ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
embeddings,
ner_model,
ner_converter
])

example = spark.createDataFrame(pd.DataFrame({'text': ['My name is John!']}))
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_mit_movie_simple_distilbert_base_cased", "en") 
.setInputCols("document"', "token", "embeddings") 
.setOutputCol("ner")

val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))
val result = pipeline.fit(Seq.empty["My name is John!"].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["My name is John!"]

ner_df = nlu.load('en.ner. ner_mit_movie_simple_distilbert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_mit_movie_simple_distilbert_base_cased|
|Type:|ner|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

[https://groups.csail.mit.edu/sls/downloads/movie/](https://groups.csail.mit.edu/sls/downloads/movie/)

## Benchmarking

```bash
processed 24686 tokens with 5339 phrases; found: 5331 phrases; correct: 4677.
accuracy:  87.88%; (non-O)
accuracy:  93.74%; precision:  87.73%; recall:  87.60%; FB1:  87.67
ACTOR: precision:  88.34%; recall:  95.20%; FB1:  91.64  875
CHARACTER: precision:  64.56%; recall:  56.67%; FB1:  60.36  79
DIRECTOR: precision:  93.00%; recall:  84.43%; FB1:  88.51  414
GENRE: precision:  91.04%; recall:  94.63%; FB1:  92.80  1161
PLOT: precision:  70.86%; recall:  72.30%; FB1:  71.57  501
RATING: precision:  93.16%; recall:  92.60%; FB1:  92.88  497
RATINGS_AVERAGE: precision:  83.94%; recall:  86.92%; FB1:  85.40  467
REVIEW: precision:  47.06%; recall:  14.29%; FB1:  21.92  17
SONG: precision:  76.32%; recall:  53.70%; FB1:  63.04  38
TITLE: precision:  84.60%; recall:  83.10%; FB1:  83.84  552
TRAILER: precision:  83.87%; recall:  86.67%; FB1:  85.25  31
YEAR: precision:  95.99%; recall:  93.19%; FB1:  94.57  699

```