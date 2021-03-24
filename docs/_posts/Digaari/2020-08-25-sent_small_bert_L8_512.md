---
layout: model
title: Smaller BERT Sentence Embeddings (L-8_H-512_A-8)
author: John Snow Labs
name: sent_small_bert_L8_512
date: 2020-08-25
task: Embeddings
language: en
edition: Spark NLP 2.6.0
tags: [open_source, embeddings, en]
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This is one of the smaller BERT models referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962).  The smaller BERT models are intended for environments with restricted computational resources. They can be fine-tuned in the same manner as the original BERT models. However, they are most effective in the context of knowledge distillation, where the fine-tuning labels are produced by a larger and more accurate teacher.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_small_bert_L8_512_en_2.6.0_2.4_1598350686215.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("sent_small_bert_L8_512", "en") \
      .setInputCols("sentence") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["I hate cancer, "Antibiotics aren't painkiller"]})))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("sent_small_bert_L8_512", "en")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val result = pipeline.fit(Seq.empty["I hate cancer, "Antibiotics aren't painkiller"].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.small_bert_L8_512').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	en_embed_sentence_small_bert_L8_512_embeddings	      sentence
		
 	[0.07683686912059784, -0.09125291556119919, 1.... 	I hate cancer
 	[0.05132533982396126, 0.16612868010997772, -0.... 	Antibiotics aren't painkiller
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_small_bert_L8_512|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|512|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1
