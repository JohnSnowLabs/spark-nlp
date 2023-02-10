---
layout: model
title: Multilingual Representations for Indian Languages (MuRIL)
author: John Snow Labs
name: bert_muril
date: 2021-08-29
tags: [embeddings, xx, open_source]
task: Embeddings
language: xx
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A BERT model pre-trained on 17 Indian languages, and their transliterated counterparts. 

This model uses a BERT base architecture [1] pretrained from scratch using the Wikipedia [2], Common Crawl [3], PMINDIA [4] and Dakshina [5] corpora for the following 17 Indian languages: 

`Assamese`, `Bengali` , `English` , `Gujarati` , `Hindi` , `Kannada` , `Kashmiri` , `Malayalam` , `Marathi` , `Nepali` , `Oriya` , `Punjabi` , `Sanskrit` , `Sindhi` , `Tamil` , `Telugu` , `Urdu`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_muril_xx_3.2.0_3.0_1630224119168.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_muril_xx_3.2.0_3.0_1630224119168.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_muril", "xx") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_muril", "xx")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.bert.muril").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_muril|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|xx|
|Case sensitive:|false|

## Data Source

[1]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.

[2]: [Wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia)

[3]: [Common Crawl](http://commoncrawl.org/the-data/)

[4]: [PMINDIA](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html)

[5]: [Dakshina](https://github.com/google-research-datasets/dakshina)

The model is imported from: https://tfhub.dev/google/MuRIL/1