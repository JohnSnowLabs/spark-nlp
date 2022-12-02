---
layout: model
title: Legal English Bert Embeddings (Small, Uncased)
author: John Snow Labs
name: bert_embeddings_legal_bert_small_uncased
date: 2022-04-11
tags: [bert, embeddings, en, open_source, legal]
task: Embeddings
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Small version of the Legal Pretrained Bert Embeddings model (uncased), uploaded to Hugging Face, adapted and imported into Spark NLP. `legal-bert-small-uncased` is a English model orginally trained by `nlpaueb`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_legal_bert_small_uncased_en_3.4.2_3.0_1649676353340.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_legal_bert_small_uncased","en") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_legal_bert_small_uncased","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.legal_bert_small_uncased").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_legal_bert_small_uncased|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|131.9 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/nlpaueb/legal-bert-small-uncased
- https://aclanthology.org/2020.findings-emnlp.261/
- https://eur-lex.europa.eu/
- https://www.legislation.gov.uk/
- https://case.law/
- https://www.sec.gov/edgar.shtml
- https://archive.org/details/legal_bert_fp
- http://nlp.cs.aueb.gr/
