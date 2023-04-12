---
layout: model
title: XLM-RoBERTa Base for Luganda (xlm_roberta_base_finetuned_luganda)
author: John Snow Labs
name: xlm_roberta_base_finetuned_luganda
date: 2021-09-29
tags: [lg, luganda, embeddings, xlm_roberta, open_source]
task: Embeddings
language: lg
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**xlm_roberta_base_finetuned_luganda** is a ** Luganda RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Luganda language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.

Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the Luganda corpus.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_luganda_lg_3.3.0_3.0_1632913890011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_luganda_lg_3.3.0_3.0_1632913890011.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_luganda", "lg") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_luganda", "lg")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("lg.embed.xlm_roberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_luganda|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|lg|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luganda](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luganda)

## Benchmarking

```bash
## Eval results on Test set (F-score, average over 5 runs)

Dataset| XLM-R F1 | lg_roberta F1
-|-|-
[MasakhaNER](https://github.com/masakhane-io/masakhane-ner) | 79.69 | 84.70

```