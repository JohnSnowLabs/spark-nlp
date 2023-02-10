---
layout: model
title: BERT Embeddings trained on Wikipedia and BooksCorpus
author: John Snow Labs
name: bert_wiki_books
date: 2021-08-30
tags: [en, bert_embeddings, wikipedia_dataset, books_corpus_dataset, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses a BERT base architecture pretrained from scratch on Wikipedia and BooksCorpus.

This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learning that improve its accuracy over the original BERT base checkpoint.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_wiki_books_en_3.2.0_3.0_1630328923814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_wiki_books_en_3.2.0_3.0_1630328923814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_wiki_books", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])

```
```scala
val embeddings = BertEmbeddings.pretrained("bert_wiki_books", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))

```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.bert.wiki_books').predict(text, output_level='token')
embeddings_df

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_wiki_books|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[1]: [Wikipedia dataset](https://dumps.wikimedia.org/)

[2]: [BooksCorpus dataset](http://yknzhu.wixsite.com/mbweb)

This Model has been imported from: https://tfhub.dev/google/experts/bert/wiki_books/2