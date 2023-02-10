---
layout: model
title: BERT Embeddings trained on Wikipedia and BooksCorpus and fine-tuned on QQP
author: John Snow Labs
name: bert_wiki_books_qqp
date: 2021-08-30
tags: [en, open_source, wikipedia_dataset, books_corpus_dataset, bert_embeddings, qqp_dataset]
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

This model uses a BERT base architecture initialized from https://tfhub.dev/google/experts/bert/wiki_books/1 and fine-tuned on QQP.

This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learnings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_wiki_books_qqp_en_3.2.0_3.0_1630322349023.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_wiki_books_qqp_en_3.2.0_3.0_1630322349023.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_wiki_books_qqp", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_wiki_books_qqp", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.bert.wiki_books_qqp').predict(text, output_level='token')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_wiki_books_qqp|
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

[3]: [Quora Question Pairs (QQP) dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

This Model has been imported from: https://tfhub.dev/google/experts/bert/wiki_books/qqp/2