---
layout: model
title: Named Entity Recognition (NER) Model in Bengali (bengaliner_cc_300d)
author: John Snow Labs
name: bengaliner_cc_300d
date: 2021-02-10
task: Named Entity Recognition
language: bn
edition: Spark NLP 2.7.3
spark_version: 2.4
tags: [open_source, bn, ner]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect 4 different types of entities in Indian text.

## Predicted Entities

`PER`, `ORG`, `LOC`, `TIME`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_BN/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bengaliner_cc_300d_bn_2.7.3_2.4_1612957259511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bengaliner_cc_300d_bn_2.7.3_2.4_1612957259511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = WordEmbeddingsModel.pretrained("bengali_cc_300d", "bn") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("embeddings")

ner = NerDLModel.pretrained("bengaliner_cc_300d", "bn") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...
pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, ner, ner_converter])
example = spark.createDataFrame([['১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
...
val embeddings = WordEmbeddingsModel.pretrained("bengali_cc_300d", "bn") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val ner = NerDLModel.pretrained("bengaliner_cc_300d", "bn")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner, ner_converter))
val data = Seq("১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন").toDF("text")
val result = pipeline.fit(data).transform(data)

```


{:.nlu-block}
```python
import nlu
nlu.load("bn.ner").predict("""১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন""")
```

</div>

## Results

```bash
+----------------------+-----------+
| ner_chunk            | label     |
+----------------------+-----------+
| ১৯৪৮ সালে             | TIME      |
| ইয়াজউদ্দিন আহম্মেদ       | PER       |
| মুন্সিগঞ্জ উচ্চ বিদ্যালয়      | ORG       |
| ১৯৫০ সালে             | TIME      |
| মুন্সিগঞ্জ হরগঙ্গা কলেজ     | ORG       |
+----------------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bengaliner_cc_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|bn|

## Data Source

This models is trained on data obtained from https://ieeexplore.ieee.org/document/8944804

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-TIME	 167	 37	 25	 0.8186275	 0.8697917	 0.8434344
B-LOC	 678	 114	 195	 0.8560606	 0.7766323	 0.81441444
I-ORG	 287	 104	 143	 0.73401535	 0.66744184	 0.6991474
B-TIME	 414	 54	 123	 0.88461536	 0.7709497	 0.8238806
I-LOC	 98	 50	 76	 0.6621622	 0.5632184	 0.6086956
I-PER	 805	 38	 55	 0.9549229	 0.93604654	 0.94539046
B-ORG	 446	 108	 225	 0.8050541	 0.6646796	 0.72816324
B-PER	 764	 48	 183	 0.9408867	 0.80675817	 0.86867535
tp: 3659 fp: 553 fn: 1025 labels: 8
Macro-average	 prec: 0.8320431, rec: 0.75693977, f1: 0.79271656
Micro-average	 prec: 0.86870843, rec: 0.78116995, f1: 0.8226169
```
