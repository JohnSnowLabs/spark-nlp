---
layout: model
title: Marathi DistilBertForMaskedLM Cased model (from DarshanDeshpande)
author: John Snow Labs
name: distilbert_embeddings_marathi
date: 2022-12-12
tags: [mr, open_source, distilbert_embeddings, distilbertformaskedlm]
task: Embeddings
language: mr
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `marathi-distilbert` is a Marathi model originally trained by `DarshanDeshpande`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_marathi_mr_4.2.4_3.0_1670865013879.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_marathi_mr_4.2.4_3.0_1670865013879.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_embeddings_marathi","mr") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)

pipeline = Pipeline(stages=[documentAssembler, tokenizer, distilbert_loaded])

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

val distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_embeddings_marathi","mr")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(false)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, distilbert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("mr.embed.distil_bert").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_marathi|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|mr|
|Size:|247.8 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/DarshanDeshpande/marathi-distilbert
- https://github.com/DarshanDeshpande
- https://www.linkedin.com/in/darshan-deshpande/
- https://github.com/Baras64
- http://​www.linkedin.com/in/harsh-abhi