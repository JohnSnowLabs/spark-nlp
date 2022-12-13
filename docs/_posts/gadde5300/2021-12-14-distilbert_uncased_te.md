---
layout: model
title: Telugu Word Embeddings (DistilBERT)
author: John Snow Labs
name: distilbert_uncased
date: 2021-12-14
tags: [open_source, embeddings, te, distilbert, telugu]
task: Embeddings
language: te
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a DistilBERT language model pre-trained on ~2 GB of the monolingual training corpus. The pre-training data was majorly taken from OSCAR. This model can be fine-tuned on various downstream tasks like text classification, POS-tagging, question-answering, etc. Embeddings from this model can also be used for feature-based training.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_uncased_te_3.1.0_3.0_1639472349482.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_uncased_te_3.1.0_3.0_1639472349482.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('document')

sentence_detector = SentenceDetector() \
.setInputCols(['document'])\
.setOutputCol('sentence')

tokenizer = Tokenizer()\
.setInputCols(['sentence']) \
.setOutputCol('token')

distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_uncased", "te"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("embeddings")\
.setCaseSensitive(False)

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, distilbert_loaded])

pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

result = pipeline_model.transform(spark.createDataFrame([['ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు.']], ["text"]))

```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentence_detector  = SentenceDetector()
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = Tokenizer()\
.setInputCols("sentence") \
.setOutputCol("token")

val distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_uncased", "te"))\
.setInputCols("sentence","token")\
.setOutputCol("embeddings")\
.setCaseSensitive(False)

val nlp_pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, distilbert_loaded))

val data = Seq("ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("te.embed.distilbert").predict("""ఆంగ్ల పఠన పేరాల యొక్క గొప్ప మూలం కోసం చూస్తున్నారా? మీరు సరైన స్థలానికి వచ్చారు.""")
```

</div>

## Results

```bash
| Token    	| Embeddings                                                           	|
|----------	|----------------------------------------------------------------------	|
| ఆంగ్ల     	| [1.3526772, 0.66792506, 0.80145407, -1.7625582, 1.222954, ...]        |
| పఠన      	| [2.0163336, -0.11855277, -0.71146804, -0.3959106, -0.63389313, ...]  	|
| పేరాల      	| [1.0630535, 0.2587409, 0.09540532, -0.048794597, 1.0124478, ...]     	|
| యొక్క      	| [1.4005142, 0.43655983, 0.5112152, -0.9843408, 0.9581941, ...]       	|
| గొప్ప      	| [1.6955082, 0.40451798, 0.8449157, -1.0998198, 0.80302626, ...]      	|
| మూలం     	| [2.0383484, 0.90390867, -0.52174926, -0.637539, 0.29188454, ...]     	|
| కోసం      	| [1.3596793, 1.0218208, 0.26274702, -0.2437865, 0.50547075, ...]      	|
| చూస్తున్నారా 	| [1.4825231, 0.6084269, 1.5597858, -1.0951629, 0.33125773, ...]       	|
| ?        	| [2.86698, -0.07081009, 0.078133255, 0.43756652, 0.05295326, ...]     	|
| మీరు      	| [1.0796824, 0.35925022, 0.51510495, -0.9841369, 0.39694318, ...]     	|
| సరైన      	| [1.1148729, -0.004858747, 0.041157544, -0.5826167, 0.24176109, ...]  	|
| స్థలానికి    	| [1.2047833, 0.119116426, -0.039619423, -0.48747823, 0.15061232, ...] 	|
| వచ్చారు    	| [1.1785411, -0.013213344, 0.14526407, -0.60479, 0.031448614, ...]    	|
| .        	| [0.76072985, -1.9430697, -1.6266187, 0.46296686, -2.2197602, ...]    	|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_uncased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|te|
|Size:|249.4 MB|
|Case sensitive:|false|
