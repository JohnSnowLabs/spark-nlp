---
layout: model
title: BERT Embeddings trained on MEDLINE/PubMed and fine-tuned on SQuAD 2.0
author: John Snow Labs
name: bert_pubmed_squad2
date: 2021-08-30
tags: [en, open_source, squad_2_dataset, medline_pubmed_dataset, bert_embeddings]
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

This model uses a BERT base architecture initialized from https://tfhub.dev/google/experts/bert/pubmed/1 and fine-tuned on SQuAD 2.0.

This is a BERT base architecture but some changes have been made to the original training and export scheme based on more recent learnings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_pubmed_squad2_en_3.2.0_3.0_1630323544592.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_pubmed_squad2_en_3.2.0_3.0_1630323544592.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_pubmed_squad2", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_pubmed_squad2", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.bert.pubmed_squad2').predict(text, output_level='token')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_pubmed_squad2|
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

[3]: [Stanford Queston Answering (SQuAD 2.0) dataset](https://rajpurkar.github.io/SQuAD-explorer/)

[4]: [MEDLINE/PubMed dataset](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

This Model has been imported from: https://tfhub.dev/google/experts/bert/pubmed/squad2/2