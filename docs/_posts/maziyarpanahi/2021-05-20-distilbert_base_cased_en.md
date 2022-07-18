---
layout: model
title: DistilBERT base model (cased)
author: John Snow Labs
name: distilbert_base_cased
date: 2021-05-20
tags: [distilbert, en, english, open_source, embeddings]
task: Embeddings
language: en
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a distilled version of the [BERT base model](https://huggingface.co/bert-base-cased). It was introduced in [this paper](https://arxiv.org/abs/1910.01108). The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation). This model is cased: it does make a difference between english and English.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_cased_en_3.1.0_2.4_1621521790187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.distilbert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_cased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/distilbert-base-cased](https://huggingface.co/distilbert-base-cased)

## Benchmarking

```bash
When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 81.5 | 87.8 | 88.2 | 90.4  | 47.2 | 85.5  | 85.6 | 60.6 |

```