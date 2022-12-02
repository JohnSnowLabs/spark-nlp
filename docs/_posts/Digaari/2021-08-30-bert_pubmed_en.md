---
layout: model
title: BERT Embeddings trained on MEDLINE/PubMed
author: John Snow Labs
name: bert_pubmed
date: 2021-08-30
tags: [en, bert_embeddings, medline_pubmed_dataset, open_source]
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

This model uses a BERT base architecture[1] pretrained from scratch on MEDLINE/PubMed

This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learnings that improve its accuracy over the original BERT base checkpoint

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_pubmed_en_3.2.0_3.0_1630316760578.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_pubmed", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_pubmed", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.bert.pubmed').predict(text, output_level='token')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_pubmed|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[1]: [MEDLINE/PubMed dataset](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

This Model has been imported from: https://tfhub.dev/google/experts/bert/pubmed/2