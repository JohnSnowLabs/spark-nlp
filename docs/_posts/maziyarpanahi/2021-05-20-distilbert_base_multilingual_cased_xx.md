---
layout: model
title: DistilBERT base multilingual model (cased)
author: John Snow Labs
name: distilbert_base_multilingual_cased
date: 2021-05-20
tags: [distilbert, embeddings, xx, multilingual, open_source]
task: Embeddings
language: xx
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a distilled version of the [BERT base multilingual model](bert-base-multilingual-cased). The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation). This model is cased: it does make a difference between english and English. The model is trained on the concatenation of Wikipedia in 104 different languages listed [here](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages).

The model has 6 layers, 768 dimension,s and 12 heads, totalizing 134M parameters (compared to 177M parameters for mBERT-base). On average DistilmBERT is twice as fast as mBERT-base.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_xx_3.1.0_2.4_1621522568093.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DistilBertEmbeddings.pretrained("distilbert_base_multilingual_cased", "xx") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_multilingual_cased", "xx")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.distilbert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_multilingual_cased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)

## Benchmarking

```bash
| Model                        | English | Spanish | Chinese | German | Arabic  | Urdu |
| :---:                        | :---:   | :---:   | :---:   | :---:  | :---:   | :---:|
| mBERT base cased (computed)  | 82.1    | 74.6    | 69.1    | 72.3   | 66.4    | 58.5 |
| mBERT base uncased (reported)| 81.4    | 74.3    | 63.8    | 70.5   | 62.1    | 58.3 |
| DistilmBERT                  | 78.2    | 69.1    | 64.0    | 66.3   | 59.1    | 54.7 |
```