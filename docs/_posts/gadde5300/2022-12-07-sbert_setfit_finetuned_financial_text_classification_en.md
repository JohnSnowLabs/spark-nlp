---
layout: model
title: English Financial BERT Sentence Embeddings( Finance, 768D)
author: John Snow Labs
name: sbert_setfit_finetuned_financial_text_classification
date: 2022-12-07
tags: [en, english, embeddings, transformer, open_source, finance, tensorflow]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Financial BERT Sentence Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `setfit-finetuned-financial-text-classification` is a English Financial model which is maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_setfit_finetuned_financial_text_classification_en_4.2.4_3.0_1670423819963.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sbert_setfit_finetuned_financial_text_classification_en_4.2.4_3.0_1670423819963.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbert_setfit_finetuned_financial_text_classification", "en")\
  .setInputCols(["sentence"])\
  .setOutputCol("sbert_embeddings")
```
```scala
val sentence_embeddings = BertSentenceEmbeddings.pretrained("sbert_setfit_finetuned_financial_text_classification", "en")
  .setInputCols("sentence")
  .setOutputCol("bert_sentence"))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed_sentence.bert.finetuned").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_setfit_finetuned_financial_text_classification|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Size:|409.0 MB|
|Case sensitive:|true|

## References

https://huggingface.co/nickmuchi/setfit-finetuned-financial-text-classification