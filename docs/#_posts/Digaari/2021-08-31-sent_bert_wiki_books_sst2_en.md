---
layout: model
title: BERT Sentence Embeddings trained on Wikipedia and BooksCorpus and fine-tuned on SST-2
author: John Snow Labs
name: sent_bert_wiki_books_sst2
date: 2021-08-31
tags: [en, open_source, sentence_embeddings, wikipedia_dataset, books_corpus_dataset, sst_2_dataset]
task: Embeddings
language: en
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses a BERT base architecture initialized from https://tfhub.dev/google/experts/bert/wiki_books/1 and fine-tuned on SST-2. This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learnings.

This model is intended to be used for a variety of English NLP tasks. The pre-training data contains more formal text and the model may not generalize to more colloquial text such as social media or messages.

This model is fine-tuned on the SST-2 and is recommended for use in sentiment analysis tasks. The fine-tuning task uses the Stanford Sentiment Treebank (SST-2) dataset to predict the sentiment in a given sentence.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_wiki_books_sst2_en_3.2.0_3.0_1630412133457.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_wiki_books_sst2", "en") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_wiki_books_sst2", "en")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
sent_embeddings_df = nlu.load('en.embed_sentence.bert.wiki_books_sst2').predict(text, output_level='sentence')
sent_embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_wiki_books_sst2|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[1]: [Wikipedia dataset](https://dumps.wikimedia.org/)

[2]: [BooksCorpus dataset](http://yknzhu.wixsite.com/mbweb)

[3]: [Stanford Sentiment Treebank (SST-2) dataset](https://nlp.stanford.edu/sentiment/index.html)

This Model has been imported from: https://tfhub.dev/google/experts/bert/wiki_books/sst2/2