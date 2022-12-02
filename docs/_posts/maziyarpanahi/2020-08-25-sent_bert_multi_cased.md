---
layout: model
title: Multilingual BERT Sentence Embeddings (Base Cased)
author: John Snow Labs
name: sent_bert_multi_cased
date: 2020-08-25
task: Embeddings
language: xx
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, xx]
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_multi_cased_xx_2.6.0_2.4_1598347692999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer', "Antibiotics aren't painkiller"]], ["text"]))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I hate cancer, "Antibiotics aren't painkiller").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('xx.embed_sentence.bert.cased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	xx_embed_sentence_bert_cased_embeddings	            sentence
		
[-0.3695415258407593, -0.33228799700737, 0.553... 	I hate cancer
	[-0.2776091396808624, -0.35221806168556213, 0.... 	Antibiotics aren't painkiller
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_multi_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[xx]|
|Dimension:|768|
|Case sensitive:|true|

{:.h2_title}
## Data Source
The model is imported from https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3
