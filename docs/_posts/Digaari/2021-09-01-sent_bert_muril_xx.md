---
layout: model
title: Multilingual Representations for Indian Languages (MuRIL) - BERT Sentence Embedding pre-trained on 17 Indian languages
author: John Snow Labs
name: sent_bert_muril
date: 2021-09-01
tags: [xx, open_source, sentence_embeddings, muril, indian_languages]
task: Embeddings
language: xx
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses a BERT base architecture pretrained from scratch using the Wikipedia, Common Crawl, PMINDIA and Dakshina corpora for the following 17 Indian languages:

`Assamese`, `Bengali` , `English` , `Gujarati` , `Hindi` , `Kannada` , `Kashmiri` , `Malayalam` , `Marathi` , `Nepali` , `Oriya` , `Punjabi` , `Sanskrit` , `Sindhi` , `Tamil` , `Telugu` , `Urdu`

The MuRIL model is pre-trained on monolingual segments as well as parallel segments as detailed below :

- Monolingual Data : Publicly available corpora from Wikipedia and Common Crawl for 17 Indian languages.
- Parallel Data : There are two types of parallel data :
- Translated Data : Translations of the above monolingual corpora obtained using the Google NMT pipeline. Translated segment pairs fed as input. Also, Publicly available PMINDIA corpus was used.
- Transliterated Data : Transliterations of Wikipedia obtained using the IndicTrans library. Transliterated segment pairs fed as input. Also, Publicly available Dakshina dataset was used.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_muril_xx_3.2.0_3.0_1630467991919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_muril", "xx") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_muril", "xx")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
sent_embeddings_df = nlu.load('en.embed_sentence.bert.muril').predict(text, output_level='sentence')
sent_embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_muril|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|xx|
|Case sensitive:|false|

## Data Source

[1]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.

[2]: [Wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia)

[3]: [Common Crawl](http://commoncrawl.org/the-data/)

[4]: [PMINDIA](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html)

[5]: [Dakshina](https://github.com/google-research-datasets/dakshina)

The model is imported from: https://tfhub.dev/google/MuRIL/1
