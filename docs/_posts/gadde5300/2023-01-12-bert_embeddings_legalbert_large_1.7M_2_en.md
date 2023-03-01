---
layout: model
title: English Legal BERT Embedding Large Cased model
author: John Snow Labs
name: bert_embeddings_legalbert_large_1.7M_2
date: 2023-01-12
tags: [en, open_source, embeddings, bert]
task: Embeddings
language: en
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legalbert-large-1.7M-2` is a English model originally trained by `pile-of-law`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_legalbert_large_1.7M_2_en_4.2.7_3.0_1673543897109.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_legalbert_large_1.7M_2_en_4.2.7_3.0_1673543897109.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_legalbert_large_1.7M_2","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_legalbert_large_1.7M_2|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Size:|648.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/pile-of-law/legalbert-large-1.7M-2
- https://github.com/LexPredict/lexpredict-lexnlp
- https://arxiv.org/abs/2110.00976
- https://arxiv.org/abs/1907.11692
- https://arxiv.org/abs/1810.04805