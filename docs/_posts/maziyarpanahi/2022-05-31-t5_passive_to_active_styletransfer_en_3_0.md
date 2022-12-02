---
layout: model
title: T5 for Passive to Active Style Transfer
author: John Snow Labs
name: t5_passive_to_active_styletransfer
date: 2022-05-31
tags: [t5, en, grammar_check, open_source]
task: Text Generation
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a text-to-text model based on T5 fine-tuned to generate passively written text from an actively written text input, for the task "transfer Passive to Active:". It is based on Prithiviraj Damodaran's Styleformer.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/T5_LINGUISTIC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_passive_to_active_styletransfer_en_4.0.0_3.0_1654004126619.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

t5 = T5Transformer.pretrained("t5_passive_to_active_styletransfer") \
.setTask("transfer Passive to Active:") \
.setInputCols(["documents"]) \
.setMaxOutputLength(200) \
.setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["A letter was sent to you."]]).toDF("text")
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

val t5 = T5Transformer.pretrained("t5_passive_to_active_styletransfer")
.setTask("transfer Passive to Active:")
.setMaxOutputLength(200)
.setInputCols("documents")
.setOutputCol("transfer")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("A letter was sent to you.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("transfer.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5.passive_to_active_styletransfer").predict("""transfer Passive to Active:""")
```

</div>

## Results

```bash
+-------------------+
|result             |
+-------------------+
|[you sent a letter]|
+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_passive_to_active_styletransfer|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|en|
|Size:|265.4 MB|

## References

The original model is from the transformers library:

https://huggingface.co/prithivida/passive_to_active_styletransfer