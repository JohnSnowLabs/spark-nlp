---
layout: model
title: Chinese Deberta Embeddings Cased model (from IDEA-CCNL)
author: John Snow Labs
name: deberta_embeddings_erlangshen_v2_chinese_sentencepiece
date: 2023-02-23
tags: [open_source, deberta, deberta_embeddings, debertav2formaskedlm, zh, tensorflow]
task: Embeddings
language: zh
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DeBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DebertaV2ForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece` is a Chinese model originally trained by `IDEA-CCNL`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_embeddings_erlangshen_v2_chinese_sentencepiece_zh_4.3.0_3.0_1677192535880.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_embeddings_erlangshen_v2_chinese_sentencepiece_zh_4.3.0_3.0_1677192535880.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_erlangshen_v2_chinese_sentencepiece","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark-NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_erlangshen_v2_chinese_sentencepiece","zh")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark-NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_embeddings_erlangshen_v2_chinese_sentencepiece|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|zh|
|Size:|445.8 MB|
|Case sensitive:|false|

## References

https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece
