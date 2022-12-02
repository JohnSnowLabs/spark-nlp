---
layout: model
title: Finnish BERT Embeddings (Base Cased)
author: John Snow Labs
name: bert_base_finnish_cased
date: 2022-01-03
tags: [open_source, embeddings, fi, bert]
task: Embeddings
language: fi
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A version of Google's BERT deep transfer learning model for Finnish. The model can be fine-tuned to achieve state-of-the-art results for various Finnish natural language processing tasks.

FinBERT features a custom 50,000 wordpiece vocabulary that has much better coverage of Finnish words than e.g. the previously released multilingual BERT models from Google.

FinBERT has been pre-trained for 1 million steps on over 3 billion tokens (24B characters) of Finnish text drawn from news, online discussion, and internet crawls. By contrast, Multilingual BERT was trained on Wikipedia texts, where the Finnish Wikipedia text is approximately 3% of the amount used to train FinBERT.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_finnish_cased_fi_3.3.4_2.4_1641223279447.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_base_finnish_cased", "fi") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

sample_data= spark.createDataFrame([['Syväoppiminen perustuu keinotekoisiin hermoihin, jotka muodostavat monikerroksisen neuroverkon.']], ["text"])
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(sample_data)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_base_finnish_cased", "fi")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Syväoppiminen perustuu keinotekoisiin hermoihin, jotka muodostavat monikerroksisen neuroverkon.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fi.embed_sentence.bert.cased").predict("""Syväoppiminen perustuu keinotekoisiin hermoihin, jotka muodostavat monikerroksisen neuroverkon.""")
```

</div>

## Results

```bash
+--------------------+---------------+
|          embeddings|          token|
+--------------------+---------------+
|[0.53366333, -0.4...|  Syväoppiminen|
|[0.49171034, -1.1...|       perustuu|
|[-0.0017492473, -...| keinotekoisiin|
|[0.61259747, -0.7...|      hermoihin|
|[-0.008151092, -0...|              ,|
|[-0.4050159, -0.2...|          jotka|
|[-0.69079936, 0.6...|    muodostavat|
|[-0.45641452, 0.4...|monikerroksisen|
|[1.278124, -1.218...|    neuroverkon|
|[0.42451048, -1.2...|              .|
+--------------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_finnish_cased|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert]|
|Language:|fi|
|Size:|464.2 MB|
|Case sensitive:|true|
