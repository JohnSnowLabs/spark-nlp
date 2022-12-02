---
layout: model
title: Smaller BERT Sentence Embeddings (L-8_H-128_A-2)
author: John Snow Labs
name: sent_small_bert_L8_128
date: 2020-08-25
task: Embeddings
language: en
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, en]
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This is one of the smaller BERT models referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962).  The smaller BERT models are intended for environments with restricted computational resources. They can be fine-tuned in the same manner as the original BERT models. However, they are most effective in the context of knowledge distillation, where the fine-tuning labels are produced by a larger and more accurate teacher.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_small_bert_L8_128_en_2.6.0_2.4_1598350334113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L8_128", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer', "Antibiotics aren't painkiller"]], ["text"]))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L8_128", "en")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I hate cancer, "Antibiotics aren't painkiller").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.small_bert_L8_128').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	en_embed_sentence_small_bert_L8_128_embeddings	      sentence
		
	[1.3234734535217285, -0.39469125866889954, 0.0... 	I hate cancer
	[1.8613495826721191, -0.7580474019050598, -0.6... 	Antibiotics aren't painkiller
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_small_bert_L8_128|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|128|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1)
