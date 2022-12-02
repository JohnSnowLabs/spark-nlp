---
layout: model
title: Multilingual BERT Embeddings (Base Cased)
author: John Snow Labs
name: bert_multi_cased
date: 2020-08-25
task: Embeddings
language: xx
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, xx]
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_multi_cased_xx_2.6.0_2.4_1598341875191.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("bert_multi_cased", "xx") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love Spark NLP']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("bert_multi_cased", "xx")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I love Spark NLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love Spark NLP"]
embeddings_df = nlu.load('xx.embed.bert_multi_cased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	xx_embed_bert_multi_cased_embeddings	            token
		
[0.31631314754486084, -0.5579454898834229, 0.1... 	I
	[-0.1488783359527588, -0.27264419198036194, -0... 	love
	[0.0496230386197567, -0.43625175952911377, -0.... 	Spark
	[-0.2838578224182129, -0.7103433012962341, 0.4... 	NLP
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_multi_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[xx]|
|Dimension:|768|
|Case sensitive:|true|

{:.h2_title}
## Data Source
The model is imported from https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3
