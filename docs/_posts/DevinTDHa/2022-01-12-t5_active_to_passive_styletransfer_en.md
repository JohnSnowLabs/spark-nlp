---
layout: model
title: T5 for Active to Passive Style Transfer
author: John Snow Labs
name: t5_active_to_passive_styletransfer
date: 2022-01-12
tags: [t5, open_source, en]
task: Text Generation
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a text-to-text model based on T5 fine-tuned to generate actively written text from a passively written text input, for the task "transfer Active to Passive:". It is based on Prithiviraj Damodaran's Styleformer.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/T5_LINGUISTIC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_active_to_passive_styletransfer_en_3.4.0_3.0_1641987400533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_active_to_passive_styletransfer_en_3.4.0_3.0_1641987400533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

spark = sparknlp.start()

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_active_to_passive_styletransfer") \
.setTask("transfer Active to Passive:") \
.setInputCols(["documents"]) \
.setMaxOutputLength(200) \
.setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["I am writing you a letter."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("transfers.result").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("documents")

val t5 = T5Transformer.pretrained("t5_active_to_passive_styletransfer")
.setTask("transfer Active to Passive:")
.setMaxOutputLength(200)
.setInputCols("documents")
.setOutputCol("transfer")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("I am writing you a letter.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("transfer.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5.active_to_passive_styletransfer").predict("""transfer Active to Passive:""")
```

</div>

## Results

```bash
+---------------------------+
|result                     |
+---------------------------+
|[a letter is written by me]|
+---------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_active_to_passive_styletransfer|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[transfers]|
|Language:|en|
|Size:|264.5 MB|

## Data Source

The original model is from the transformers library:

https://huggingface.co/prithivida/active_to_passive_styletransfer
