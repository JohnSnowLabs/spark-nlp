---
layout: model
title: XLM-RoBERTa Base for Amharic (xlm_roberta_base_finetuned_amharic)
author: John Snow Labs
name: xlm_roberta_base_finetuned_amharic
date: 2021-09-29
tags: [open_source, xlm_roberta, embeddings, am, amharic]
task: Embeddings
language: am
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**xlm_roberta_base_finetuned_amharic** is a **Amharic RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Amharic language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.
  
Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the Amharic corpus.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_amharic_am_3.3.0_3.0_1632912241061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_amharic", "am") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```
```scala
val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_amharic", "am")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("am.embed.xlm_roberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_amharic|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|am|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/Davlan/xlm-roberta-base-finetuned-amharic](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-amharic)

## Benchmarking

```bash
## Eval results on the Test set (F-score, average over 5 runs)

Dataset| XLM-R F1 | am_roberta F1
-|-|-
[MasakhaNER](https://github.com/masakhane-io/masakhane-ner) | 70.96 | 77.97
```