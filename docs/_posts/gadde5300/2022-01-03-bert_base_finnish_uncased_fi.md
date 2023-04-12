---
layout: model
title: Finnish BERT Embeddings (Base Uncased)
author: John Snow Labs
name: bert_base_finnish_uncased
date: 2022-01-03
tags: [open_source, embeddings, fi, bert]
task: Embeddings
language: fi
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: BertSentenceEmbeddings
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_finnish_uncased_fi_3.4.0_3.0_1641223281610.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_finnish_uncased_fi_3.4.0_3.0_1641223281610.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_base_finnish_uncased", "fi") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

sample_data= spark.createDataFrame([['Syväoppimisen tavoitteena on luoda algoritmien avulla neuroverkko, joka pystyy ratkaisemaan sille annetut ongelmat.']], ["text"])
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

val embeddings = BertEmbeddings.pretrained("bert_base_finnish_uncased", "fi")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Syväoppimisen tavoitteena on luoda algoritmien avulla neuroverkko, joka pystyy ratkaisemaan sille annetut ongelmat.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fi.embed_sentence.bert").predict("""Syväoppimisen tavoitteena on luoda algoritmien avulla neuroverkko, joka pystyy ratkaisemaan sille annetut ongelmat.""")
```

</div>

## Results

```bash
+--------------------+-------------+
|          embeddings|        token|
+--------------------+-------------+
|[0.9422476, -0.14...|Syväoppimisen|
|[2.0408847, -1.45...|  tavoitteena|
|[2.33223, -1.7228...|           on|
|[0.6425015, -0.96...|        luoda|
|[0.10455999, -0.2...|  algoritmien|
|[0.28626734, -0.2...|       avulla|
|[1.0091506, -0.75...|  neuroverkko|
|[1.501086, -0.651...|            ,|
|[1.2654709, -0.82...|         joka|
|[1.710053, -0.406...|       pystyy|
|[0.43736708, -0.2...| ratkaisemaan|
|[1.0496894, 0.191...|        sille|
|[0.8630942, -0.16...|      annetut|
|[0.50174934, -1.3...|     ongelmat|
|[0.27278847, -0.9...|            .|
+--------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_finnish_uncased|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert]|
|Language:|fi|
|Size:|464.1 MB|
|Case sensitive:|false|
