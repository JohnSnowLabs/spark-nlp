---
layout: model
title: "MedEmbed base: Specialized Embedding Model for Medical and Clinical Information Retrieval (OpenVINO)"
author: John Snow Labs
name: bge_medembed_base_v0_1_openvino
date: 2025-07-15
tags: [openvino, english, medical_embedding, clinical_embedding, information_retrieval, open_source, bge, en]
task: Embeddings
language: en
edition: Spark NLP 6.0.0
spark_version: 3.0
supported: true
engine: openvino
annotator: BGEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

MedEmbed is a family of embedding models fine-tuned specifically for medical and clinical data, designed to enhance performance in healthcare-related natural language processing (NLP) tasks, particularly information retrieval.

GitHub Repo: https://github.com/abhinand5/MedEmbed
Technical Blog Post: https://huggingface.co/blog/abhinand/medembed-finetuned-embedding-models-for-medical-ir

This model is intended for use in medical and clinical contexts to improve information retrieval, question answering, and semantic search tasks. It can be integrated into healthcare systems, research tools, and medical literature databases to enhance search capabilities and information access.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_medembed_base_v0_1_openvino_en_6.0.0_3.0_1752605366919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_medembed_base_v0_1_openvino_en_6.0.0_3.0_1752605366919.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BGEEmbeddings
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

bge_loaded = BGEEmbeddings.load("bge_medembed_base_v0_1_openvino")\
    .setInputCols(["document"])\
    .setOutputCol("embeddings")\

pipeline = Pipeline(
    stages = [
        document_assembler,
        bge_loaded
  ])

data = spark.createDataFrame([
    ['William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist.']
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr("explode(embeddings.embeddings) as embeddings").show()

```
```scala
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.explode
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val bertEmbeddings = BertSentenceEmbeddings.load("bge_medembed_base_v0_1_openvino")
  .setInputCols("document")
  .setOutputCol("bert")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  bertEmbeddings
))

val data = Seq(
  "William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist."
).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select(explode($"bert.embeddings").alias("embeddings")).show(false)

```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[-0.055220805, 0....|
+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_medembed_base_v0_1_openvino|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|389.7 MB|