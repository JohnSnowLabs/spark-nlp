---
layout: model
title: XLM-RoBERTa Base for Wolof (xlm_roberta_base_finetuned_wolof)
author: John Snow Labs
name: xlm_roberta_base_finetuned_wolof
date: 2021-10-16
tags: [open_source, xlm_roberta, embeddings, wolof, wo]
task: Embeddings
language: wo
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**xlm_roberta_base_finetuned_wolof** is a **Wolof RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Wolof language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.

Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the **Wolof** corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_wolof_wo_3.3.1_3.0_1634413017686.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document = DocumentAssembler()\ 
.setInputCol("text")\ 
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
.setInputCols(["document"])\ 
.setOutputCol("sentence")

tokenizer = Tokenizer()\ 
.setInputCols(["sentence"])\ 
.setOutputCol("token") 

embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_wolof", "wo")\ 
.setInputCols(["sentence", "token"])\ 
.setOutputCol("embeddings")

```
```scala

val document = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = new Tokenizer() 
.setInputCols("sentence") 
.setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_wolof", "wo")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("wo.embed.xlm_roberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_wolof|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|wo|
|Case sensitive:|true|

## Data Source

Model is trained by [David Adelani](https://huggingface.co/Davlan)

Improted from [https://huggingface.co/Davlan/xlm-roberta-base-finetuned-wolof](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-wolof)