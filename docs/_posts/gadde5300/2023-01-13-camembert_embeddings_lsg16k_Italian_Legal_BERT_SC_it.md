---
layout: model
title: Italian Legal CamemBERT Embedding  Cased model
author: John Snow Labs
name: camembert_embeddings_lsg16k_Italian_Legal_BERT_SC
date: 2023-01-13
tags: [it, open_source, embeddings, camembert]
task: Embeddings
language: it
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `lsg16k-Italian-Legal-BERT-SC` is a Italian model originally trained by `dlicari`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_embeddings_lsg16k_Italian_Legal_BERT_SC_it_4.2.7_3.0_1673597331922.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_embeddings_lsg16k_Italian_Legal_BERT_SC_it_4.2.7_3.0_1673597331922.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = CamemBertEmbeddings.pretrained("camembert_embeddings_lsg16k_Italian_Legal_BERT_SC","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_embeddings_lsg16k_Italian_Legal_BERT_SC|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|it|
|Size:|460.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/dlicari/lsg16k-Italian-Legal-BERT-SC