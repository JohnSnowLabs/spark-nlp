---
layout: model
title: Multilingual BioLORD-2023-M XlmRoBertaSentenceEmbeddings from FremyCompany
author: John Snow Labs
name: sent_xlm_roberta_biolord_2023_m
date: 2025-02-14
tags: [multilingual, sentence_embeddings, xlm_roberta, open_source, xx, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.2
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained `XlmRoBertaSentenceEmbeddings` model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `sent_xlm_roberta_biolord_2023_m` is a multilingual model originally trained by FremyCompany. It supports English, Spanish, French, German, Dutch, Danish and Swedish.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_biolord_2023_m_xx_5.5.2_3.0_1739548358592.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_biolord_2023_m_xx_5.5.2_3.0_1739548358592.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

embeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_biolord_2023_m","xx") \
      .setInputCols(["document"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, embeddings])

data = spark.createDataFrame([["Disfruto trabajando con Spark-NLP."]]).toDF("text")
pipelineModel = pipeline.fit(data)
result = pipelineModel.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = XlmRoBertaSentenceEmbeddings
  .pretrained("sent_xlm_roberta_biolord_2023_m", "xx")
  .setInputCols(Array("document"))
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))


val data = Seq("Disfruto trabajando con Spark-NLP.").toDF("text")

val pipelineModel = pipeline.fit(data)
val result = pipelineModel.transform(data)
```
</div>

## Results

```bash
+----------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|                              text|                                                              document|                                                   sentence_embeddings|
+----------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|Disfruto trabajando con Spark-NLP.|[{document, 0, 33, Disfruto trabajando con Spark-NLP., {sentence ->...|[{sentence_embeddings, 0, 33, Disfruto trabajando con Spark-NLP., {...|
+----------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_xlm_roberta_biolord_2023_m|
|Compatibility:|Spark NLP 5.5.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[xlm_sentence_embeddings]|
|Language:|xx|
|Size:|1.0 GB|

## References

https://huggingface.co/FremyCompany/BioLORD-2023-M