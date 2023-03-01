---
layout: model
title: English Legal BERT Embedding Small Uncased model
author: John Snow Labs
name: bert_embeddings_legal_bert_small_uncased
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

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-bert-small-uncased` is a English model originally trained by `nlpaueb`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_legal_bert_small_uncased_en_4.2.7_3.0_1673543580768.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_legal_bert_small_uncased_en_4.2.7_3.0_1673543580768.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_legal_bert_small_uncased","en") \
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
|Model Name:|bert_embeddings_legal_bert_small_uncased|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Size:|131.8 MB|
|Case sensitive:|false|
|Max sentence length:|128|

## References

- https://huggingface.co/nlpaueb/legal-bert-small-uncased
- https://www.sec.gov/edgar.shtml
- https://twitter.com/KiddoThe2B
- http://nlp.cs.aueb.gr
- https://archive.org/details/legal_bert_fp
- https://aclanthology.org/2020.findings-emnlp.261
- http://hudoc.echr.coe.int/eng
- http://www.legislation.gov.uk
- https://www.tensorflow.org/tfrc
- https://edu.google.com/programs/credits/research
- https://case.law
- https://iliaschalkidis.github.io
- https://github.com/iliaschalkidis
- https://github.com/google-research/bert
- http://eur-lex.europa.eu