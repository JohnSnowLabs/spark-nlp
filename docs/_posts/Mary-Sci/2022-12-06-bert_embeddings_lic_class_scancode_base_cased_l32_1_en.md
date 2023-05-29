---
layout: model
title: English BertForMaskedLM Base Cased model (from ayansinha)
author: John Snow Labs
name: bert_embeddings_lic_class_scancode_base_cased_l32_1
date: 2022-12-06
tags: [en, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `lic-class-scancode-bert-base-cased-L32-1` is a English model originally trained by `ayansinha`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_lic_class_scancode_base_cased_l32_1_en_4.2.4_3.0_1670326834348.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_lic_class_scancode_base_cased_l32_1_en_4.2.4_3.0_1670326834348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_lic_class_scancode_base_cased_l32_1","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert_loaded])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_lic_class_scancode_base_cased_l32_1","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.bert.cased_base.by_ayansinha").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_lic_class_scancode_base_cased_l32_1|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|406.4 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/ayansinha/lic-class-scancode-bert-base-cased-L32-1
- https://github.com/nexB/scancode-results-analyzer
- https://github.com/nexB/scancode-results-analyzer
- https://github.com/nexB/scancode-results-analyzer#quickstart---local-machine
- https://github.com/nexB/scancode-results-analyzer/blob/master/src/results_analyze/nlp_models.py
- https://github.com/nexB/scancode-results-analyzer/blob/master/src/results_analyze/nlp_models.py