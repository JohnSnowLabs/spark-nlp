---
layout: model
title: Hebrew BERT Embedding  Cased model
author: John Snow Labs
name: bert_embeddings_Legal_heBERT_ft
date: 2023-01-12
tags: [he, open_source, embeddings, bert]
task: Embeddings
language: he
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Legal-heBERT_ft` is a Hebrew model originally trained by `avichr`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_Legal_heBERT_ft_he_4.2.7_3.0_1673544289014.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_Legal_heBERT_ft_he_4.2.7_3.0_1673544289014.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_Legal_heBERT_ft","he") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["אני אוהב את Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_Legal_heBERT_ft|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|he|
|Size:|410.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/avichr/Legal-heBERT_ft
- https://github.com/avichaychriqui/HeBERT
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4147127
- https://arxiv.org/abs/1911.03090
- https://arxiv.org/abs/2010.02559