---
layout: model
title: Portuguese BERT Sentence Embedding Large Cased model (mlm-v0.11-nli-sts-v1)
author: John Snow Labs
name: sbert_bert_large_portuguese_cased_legal_mlm_v0.11_nli_sts_v1
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

Pretrained BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-portuguese-cased-legal-mlm-v0.11-nli-sts-v1` is a  model originally trained by `stjiris`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_bert_large_portuguese_cased_legal_mlm_v0.11_nli_sts_v1_pt_4.2.4_3.0_1670675199844.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = BertSentenceEmbeddings.pretrained("sbert_bert_large_portuguese_cased_legal_mlm_v0.11_nli_sts_v1","pt") \
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
|Model Name:|sbert_bert_large_portuguese_cased_legal_mlm_v0.11_nli_sts_v1|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|pt|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/stjiris/bert-large-portuguese-cased-legal-mlm-v0.11-nli-sts-v1
- https://www.SBERT.net