---
layout: model
title: Portuguese BERT Embedding Large Cased model
author: John Snow Labs
name: bert_embeddings_bert_large_portuguese_cased_legal_mlm
date: 2023-01-13
tags: [pt, open_source, embeddings, bert]
task: Embeddings
language: pt
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-portuguese-cased-legal-mlm` is a Portuguese model originally trained by `stjiris`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_large_portuguese_cased_legal_mlm_pt_4.2.7_3.0_1673598653906.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_large_portuguese_cased_legal_mlm","pt") \
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
|Model Name:|bert_embeddings_bert_large_portuguese_cased_legal_mlm|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|pt|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/stjiris/bert-large-portuguese-cased-legal-mlm
- https://www.SBERT.net
- https://rufimelo99.github.io/SemanticSearchSystemForSTJ/_static/logo.png
- https://github.com/rufimelo99
- https://www.inesc-id.pt/projects/PR07005/
- https://www.inesc-id.pt/wp-content/uploads/2019/06/INESC-ID-logo_01.png
- https://rufimelo99.github.io/SemanticSearchSystemForSTJ/