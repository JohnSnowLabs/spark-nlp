---
layout: model
title: XLM-RoBERTa Base for Hausa (xlm_roberta_base_finetuned_hausa)
author: John Snow Labs
name: xlm_roberta_base_finetuned_hausa
date: 2021-09-29
tags: [embeddings, xlm_roberta, open_source, ha, hausa]
task: Embeddings
language: ha
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**xlm_roberta_base_finetuned_hausa** is a ** Hausa RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Hausa language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.

Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the Hausa corpus.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_hausa_ha_3.3.0_3.0_1632912619448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_hausa", "ha") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_hausa", "ha)
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("ha.embed.xlm_roberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_hausa|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|ha|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/Davlan/xlm-roberta-base-finetuned-hausa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-hausa)

## Benchmarking

```bash
## Eval results on the Test set (F-score, average over 5 runs)

Dataset| XLM-R F1 | ha_roberta F1
-|-|-
[MasakhaNER](https://github.com/masakhane-io/masakhane-ner) | 86.10 | 91.47
[VOA Hausa Textclass](https://huggingface.co/datasets/hausa_voa_topics) | | 

```
