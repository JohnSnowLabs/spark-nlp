---
layout: model
title: DistilRoBERTa base model
author: John Snow Labs
name: distilroberta_base
date: 2021-05-20
tags: [roberta, embeddings, en, english, open_source]
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

This model is a distilled version of the [RoBERTa-base model](https://huggingface.co/roberta-base). It follows the same training procedure as [DistilBERT](https://huggingface.co/distilbert-base-uncased).

The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation). This model is case-sensitive: it makes a difference between english and English.

The model has 6 layers, 768 dimensions, and 12 heads, totalizing 82M parameters (compared to 125M parameters for RoBERTa-base).
On average DistilRoBERTa is twice as fast as Roberta-base.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilroberta_base_en_3.1.0_2.4_1621523016677.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilroberta_base_en_3.1.0_2.4_1621523016677.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = RoBertaEmbeddings.pretrained("distilroberta_base", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = RoBertaEmbeddings.pretrained("distilroberta_base", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.distilroberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilroberta_base|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/distilroberta-base](https://huggingface.co/distilroberta-base)

## Benchmarking

```bash
When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 84.0 | 89.4 | 90.8 | 92.5  | 59.3 | 88.3  | 86.6 | 67.9 |

```