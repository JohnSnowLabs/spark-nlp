---
layout: model
title: English Legal BERT Embedding Base Uncased model
author: John Snow Labs
name: bert_embeddings_legal_bert_base_uncased_finetuned_ledgarscotus7
date: 2023-01-12
tags: [en, open_source, embeddings, bert]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-bert-base-uncased-finetuned-ledgarscotus7` is a English model originally trained by `hatemestinbejaia`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_legal_bert_base_uncased_finetuned_ledgarscotus7_en_4.2.7_3.0_1673544252940.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_legal_bert_base_uncased_finetuned_ledgarscotus7_en_4.2.7_3.0_1673544252940.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_legal_bert_base_uncased_finetuned_ledgarscotus7","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.embed.bert.uncased_base_finetuned_legal.by_hatemestinbejaia").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_legal_bert_base_uncased_finetuned_ledgarscotus7|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Size:|409.9 MB|
|Case sensitive:|false|
|Max sentence length:|128|

## References

- https://huggingface.co/hatemestinbejaia/legal-bert-base-uncased-finetuned-ledgarscotus7