---
layout: model
title: Detect Movie Entities - MIT Movie Complex (ner_mit_movie_complex_distilbert_base_cased)
author: John Snow Labs
name: ner_mit_movie_complex_distilbert_base_cased
date: 2021-07-20
tags: [open_source, distilbert, en, english, ner, movie]
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

This NER model was trained over the MIT Movie Corpus complex queries dataset to detect movie trivia. We used DistilBertEmbeddings (distilbert_base_cased) model for the embeddings to train this NER model.

## Predicted Entities

- Actor
- Award
- Character_Name
- Director
- Genre
- Opinion
- Origin
- Plot
- Quote
- Relationship
- Soundtrack
- Year

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_mit_movie_complex_distilbert_base_cased_en_3.1.3_2.4_1626777800973.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_mit_movie_complex_distilbert_base_cased_en_3.1.3_2.4_1626777800973.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = NerDLModel.pretrained('ner_mit_movie_complex_distilbert_base_cased', 'en') \
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

val ner_model = NerDLModel.pretrained("ner_mit_movie_complex_distilbert_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_mit_movie_complex_distilbert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_mit_movie_complex_distilbert_base_cased|
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
processed 15904 tokens with 2278 phrases; found: 2277 phrases; correct: 1674.
accuracy:  89.18%; (non-O)
accuracy:  88.41%; precision:  73.52%; recall:  73.49%; FB1:  73.50
Actor: precision:  96.50%; recall:  96.13%; FB1:  96.32  515
Award: precision:  51.85%; recall:  41.18%; FB1:  45.90  27
Character_Name: precision:  72.53%; recall:  74.16%; FB1:  73.33  91
Director: precision:  81.77%; recall:  87.71%; FB1:  84.64  192
Genre: precision:  75.00%; recall:  74.54%; FB1:  74.77  324
Opinion: precision:  41.94%; recall:  48.15%; FB1:  44.83  93
Origin: precision:  37.70%; recall:  32.39%; FB1:  34.85  61
Plot: precision:  53.43%; recall:  53.60%; FB1:  53.51  627
Quote: precision:  56.25%; recall:  39.13%; FB1:  46.15  16
Relationship: precision:  55.10%; recall:  56.25%; FB1:  55.67  49
Soundtrack: precision:  42.86%; recall:  42.86%; FB1:  42.86  7
Year: precision:  94.91%; recall:  93.88%; FB1:  94.39  275
```