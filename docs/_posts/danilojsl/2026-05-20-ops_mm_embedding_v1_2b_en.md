---
layout: model
title: OpsMM Multimodal Bi-Encoder Embeddings 1.2B
author: John Snow Labs
name: ops_mm_embedding_v1_2b
date: 2026-05-20
tags: [embeddings, multimodal, image_embeddings, text_embeddings, cross_modal_retrieval, semantic_search, rag, onnx, opsmm, biencoder, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 6.4.1
spark_version: 3.4
supported: true
engine: onnx
annotator: BiEncoderMultimodalEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model generates aligned text and image embeddings for multimodal retrieval workflows. It is used with BiEncoderMultimodalEmbeddings, which consumes paired DOCUMENT and IMAGE  annotations and produces two embedding columns: one for the text side and one for the image side. The embeddings can be indexed in a vector database and used for text-to-image, image-to-text, text-to-text, or image-to-image retrieval and RAG pipelines.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/retrieval-augmented-generation/BiEncoderMultimodalEmbeddings_OpsMM_Pinec){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ops_mm_embedding_v1_2b_en_6.4.1_3.4_1779290518035.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ops_mm_embedding_v1_2b_en_6.4.1_3.4_1779290518035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use BiEncoderMultimodalEmbeddings.pretrained("ops_mm_embedding_v1_2b", "en") with a DOCUMENT input column and an IMAGE input column. The model outputs SENTENCE_EMBEDDINGS in derived  columns named `<outputCol>_doc_embeddings` and `<outputCol>_image_embeddings`.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
  from pyspark.ml import Pipeline
  from sparknlp.reader import ReaderAssembler, LayoutAlignerForVision
  from sparknlp.annotator import BiEncoderMultimodalEmbeddings

  reader = (
      ReaderAssembler()
      .setContentPath("file:///path/to/document.html")
      .setContentType("text/html")
      .setOutputCol("reader")
      .setOutputAsDocument(False)
  )

  vision_aligner = (
      LayoutAlignerForVision()
      .setInputCols(["reader_text", "reader_image"])
      .setOutputCol("vision_pair")
      .setExplodeDocs(True)
      .setAddNeighborText(True)
  )

  opsmm = (
      BiEncoderMultimodalEmbeddings.pretrained("ops_mm_embedding_v1_2b", "en")
      .setInputCols(["vision_pair_doc", "vision_pair_image"])
      .setOutputCol("opsmm")
      .setBatchSize(1)
  )

  pipeline = Pipeline(stages=[reader, vision_aligner, opsmm])
  result = pipeline.fit(spark.emptyDataFrame).transform(spark.emptyDataFrame)

  result.select("opsmm_doc_embeddings", "opsmm_image_embeddings").show(truncate=False)
```
```scala
  from pyspark.ml import Pipeline
  from sparknlp.reader import ReaderAssembler, LayoutAlignerForVision
  from sparknlp.annotator import BiEncoderMultimodalEmbeddings

  reader = (
      ReaderAssembler()
      .setContentPath("file:///path/to/document.html")
      .setContentType("text/html")
      .setOutputCol("reader")
      .setOutputAsDocument(False)
  )

  vision_aligner = (
      LayoutAlignerForVision()
      .setInputCols(["reader_text", "reader_image"])
      .setOutputCol("vision_pair")
      .setExplodeDocs(True)
      .setAddNeighborText(True)
  )

  opsmm = (
      BiEncoderMultimodalEmbeddings.pretrained("ops_mm_embedding_v1_2b", "en")
      .setInputCols(["vision_pair_doc", "vision_pair_image"])
      .setOutputCol("opsmm")
      .setBatchSize(1)
  )

  pipeline = Pipeline(stages=[reader, vision_aligner, opsmm])
  result = pipeline.fit(spark.emptyDataFrame).transform(spark.emptyDataFrame)

  result.select("opsmm_doc_embeddings", "opsmm_image_embeddings").show(truncate=False)
```
</div>

## Results

```bash
 The model produces 1536-dimensional embeddings for both text and image inputs. It does not produce labels, entities, or generated text
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ops_mm_embedding_v1_2b|
|Compatibility:|Spark NLP 6.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[vision_pair_doc, vision_pair_image]|
|Output Labels:|[mm]|
|Language:|en|
|Size:|3.0 GB|