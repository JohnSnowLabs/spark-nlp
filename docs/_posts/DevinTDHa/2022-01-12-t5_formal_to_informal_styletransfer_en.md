---
layout: model
title: T5 for Formal to Informal Style Transfer
author: John Snow Labs
name: t5_formal_to_informal_styletransfer
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

This is a text-to-text model based on T5 fine-tuned to generate informal text from a formal text input, for the task "transfer Formal to Casual:". It is based on Prithiviraj Damodaran's Styleformer.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/T5_LINGUISTIC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_formal_to_informal_styletransfer_en_3.4.0_3.0_1641984515976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
t5 = T5Transformer.pretrained("t5_formal_to_informal_styletransfer") \
.setTask("transfer Formal to Casual:") \
.setInputCols(["documents"]) \
.setMaxOutputLength(200) \
.setOutputCol("transfers")
pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["Please leave the room now."]]).toDF("text")
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
val t5 = T5Transformer.pretrained("t5_formal_to_informal_styletransfer")
.setTask("transfer Formal to Casual:")
.setMaxOutputLength(200)
.setInputCols("documents")
.setOutputCol("transfers")
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
val data = Seq("Please leave the room now.")
.toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("transfers.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5.formal_to_informal_styletransfer").predict("""transfer Formal to Casual:""")
```

</div>

## Results

```bash
+---------------------+
|result               |
+---------------------+
|[leave the room now.]|
+---------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_formal_to_informal_styletransfer|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[transfers]|
|Language:|en|
|Size:|923.9 MB|

## Data Source

The original model is from the transformers library:

https://huggingface.co/prithivida/formal_to_informal_styletransfer