---
layout: model
title: German Financial Bert Word Embeddings
author: John Snow Labs
name: bert_sentence_embeddings_financial
date: 2022-05-04
tags: [bert, embeddings, de, open_source, financial]
task: Embeddings
language: de
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Although in the name of the model you will see the word `sentence`, this is a Word Embeddings Model.

Financial Pretrained BERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `german-financial-statements-bert` is a German model orginally trained by `fabianrausch`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sentence_embeddings_financial_de_3.4.2_3.0_1651678415089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_sentence_embeddings_financial","de") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Ich liebe Spark-NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
  .setInputCol("text") 
  .setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_sentence_embeddings_financial","de") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Ich liebe Spark-NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sentence_embeddings_financial|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|de|
|Size:|409.6 MB|
|Case sensitive:|true|
