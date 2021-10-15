---
layout: model
title: XLM-RoBERTa Base for Kinyarwanda (xlm_roberta_base_finetuned_kinyarwanda)
author: John Snow Labs
name: xlm_roberta_base_finetuned_kinyarwanda
date: 2021-10-15
tags: [open_source, xlm_roberta, embeddings, kinyarwanda, rw]
task: Embeddings
language: rw
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**xlm_roberta_base_finetuned_kinyarwanda** is a **Kinyarwanda RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Kinyarwanda language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.
            
Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the **Kinyarwanda** corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_kinyarwanda_rw_3.3.1_3.0_1634303330905.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
                
documentAssembler = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("document")

sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
    .setInputCols(["document"])\ 
    .setOutputCol("sentence")

marian = XlmRoBertaSentenceEmbeddings.pretrained("xlm_roberta_base_finetuned_kinyarwanda", "rw")\ 
    .setInputCols(["sentence"])\ 
    .setOutputCol("sentence_embeddings")

```
```scala

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
  .setInputCols("document")
  .setOutputCol("sentence")

val marian = XlmRoBertaSentenceEmbeddings.pretrained("xlm_roberta_base_finetuned_kinyarwanda", "rw")
    .setInputCols("sentence")
    .setOutputCol("sentence_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_kinyarwanda|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|rw|
|Case sensitive:|true|

## Data Source

Model is trained by [David Adelani](https://huggingface.co/Davlan)

Improted from [https://huggingface.co/Davlan/xlm-roberta-base-finetuned-kinyarwanda](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-kinyarwanda)