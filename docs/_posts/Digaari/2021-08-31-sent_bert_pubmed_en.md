---
layout: model
title: BERT Sentence Embeddings trained on MEDLINE/PubMed
author: John Snow Labs
name: sent_bert_pubmed
date: 2021-08-31
tags: [en, open_source, sentence_embeddings, medline_pubmed_dataset]
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

This model uses a BERT base architecture pretrained from scratch on MEDLINE/PubMed. This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learnings that improve its accuracy over the original BERT base checkpoint.

This model is intended to be used for a variety of English NLP tasks in the medical domain. The pre-training data contains more medical text and the model may not generalize to text outside of that domain.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_pubmed_en_3.2.0_3.0_1630412084893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_pubmed_en_3.2.0_3.0_1630412084893.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_pubmed", "en") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_pubmed", "en")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
sent_embeddings_df = nlu.load('en.embed_sentence.bert.pubmed').predict(text, output_level='sentence')
sent_embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_pubmed|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[1]: [MEDLINE/PubMed dataset](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

This Model has been imported from: https://tfhub.dev/google/experts/bert/pubmed/2