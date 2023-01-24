---
layout: model
title: Detect Movie Entities - MIT Movie Complex (ner_mit_movie_complex_bert_base_cased)
author: John Snow Labs
name: ner_mit_movie_complex_bert_base_cased
date: 2021-07-20
tags: [open_source, bert, en, english, ner, movie]
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

This NER model was trained over the MIT Movie Corpus complex queries dataset to detect movie trivia. We used BertEmbeddings (bert_base_cased) model for the embeddings to train this NER model.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_mit_movie_complex_bert_base_cased_en_3.1.3_2.4_1626776981448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_mit_movie_complex_bert_base_cased_en_3.1.3_2.4_1626776981448.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings\
.pretrained('bert_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_mit_movie_complex_bert_base_cased', 'en') \
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

val embeddings = BertEmbeddings.pretrained("bert_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_mit_movie_complex_bert_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_mit_movie_complex_bert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_mit_movie_complex_bert_base_cased|
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
processed 15904 tokens with 2278 phrases; found: 2292 phrases; correct: 1664.
accuracy:  88.81%; (non-O)
accuracy:  88.78%; precision:  72.60%; recall:  73.05%; FB1:  72.82
Actor: precision:  96.46%; recall:  94.97%; FB1:  95.71  509
Award: precision:  63.64%; recall:  61.76%; FB1:  62.69  33
Character_Name: precision:  61.62%; recall:  68.54%; FB1:  64.89  99
Director: precision:  83.43%; recall:  84.36%; FB1:  83.89  181
Genre: precision:  74.07%; recall:  73.62%; FB1:  73.85  324
Opinion: precision:  39.18%; recall:  46.91%; FB1:  42.70  97
Origin: precision:  35.37%; recall:  40.85%; FB1:  37.91  82
Plot: precision:  53.95%; recall:  53.60%; FB1:  53.77  621
Quote: precision:  64.29%; recall:  39.13%; FB1:  48.65  14
Relationship: precision:  48.00%; recall:  50.00%; FB1:  48.98  50
Soundtrack: precision:  80.00%; recall:  57.14%; FB1:  66.67  5
Year: precision:  94.22%; recall:  93.88%; FB1:  94.05  277
```
