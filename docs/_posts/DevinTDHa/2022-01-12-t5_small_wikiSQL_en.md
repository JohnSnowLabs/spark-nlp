---
layout: model
title: T5-small fine-tuned on WikiSQL
author: John Snow Labs
name: t5_small_wikiSQL
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

Google's T5 small fine-tuned on WikiSQL for English to SQL translation. Will generate SQL code from natural language input when task is set it to "translate English to SQL:".

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/T5_SQL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_SQL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_small_wikiSQL_en_3.4.0_3.0_1641982554211.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
t5 = T5Transformer.pretrained("t5_small_wikiSQL") \
.setTask("translate English to SQL:") \
.setInputCols(["documents"]) \
.setMaxOutputLength(200) \
.setOutputCol("sql")
pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["How many customers have ordered more than 2 items?"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("sql.result").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import org.apache.spark.ml.Pipeline
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("documents")
val t5 = T5Transformer.pretrained("t5_small_wikiSQL")
.setInputCols("documents")
.setMaxOutputLength(200)
.setOutputCol("sql")
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
val data = Seq("How many customers have ordered more than 2 items?")
.toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("sql.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5.wikiSQL").predict("""How many customers have ordered more than 2 items?""")
```

</div>

## Results

```bash
+----------------------------------------------------+
|result                                              |
+----------------------------------------------------+
|[SELECT COUNT Customers FROM table WHERE Orders > 2]|
+----------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_small_wikiSQL|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[sql]|
|Language:|en|
|Size:|262.1 MB|

## Data Source

Model originally from the transformer model of Manuel Romero/mrm8488.
https://huggingface.co/mrm8488/t5-small-finetuned-wikiSQL