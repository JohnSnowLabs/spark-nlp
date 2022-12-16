---
layout: model
title: Portuguese BERT Sentence Embedding Base Cased model (sts-base-ma-v2)
author: John Snow Labs
name: sbert_Legal_BERTimbau_sts_base_ma_v2
date: 2022-12-10
tags: [pt, open_source, embeddings, bert]
task: Embeddings
language: pt
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Legal-BERTimbau-sts-base-ma-v2` is a  model originally trained by `rufimelo`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_Legal_BERTimbau_sts_base_ma_v2_pt_4.2.4_3.0_1670670553618.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")
  
embeddings = BertSentenceEmbeddings.pretrained("sbert_Legal_BERTimbau_sts_base_ma_v2","pt") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Eu amo Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_Legal_BERTimbau_sts_base_ma_v2|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|pt|
|Size:|408.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/rufimelo/Legal-BERTimbau-sts-base-ma-v2
- https://www.SBERT.net
