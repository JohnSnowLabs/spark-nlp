---
layout: model
title: Finnish BERT Sentence Embeddings (Base Uncased)
author: John Snow Labs
name: sent_bert_finnish_uncased
date: 2020-08-31
task: Embeddings
language: fi
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, fi]
supported: true
deprecated: true
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A version of Google's BERT deep transfer learning model for Finnish. The model can be fine-tuned to achieve state-of-the-art results for various Finnish natural language processing tasks. `FinBERT` features a custom 50,000 wordpiece vocabulary that has much better coverage of Finnish words.

`FinBERT` has been pre-trained for 1 million steps on over 3 billion tokens (24B characters) of Finnish text drawn from news, online discussion, and internet crawls. By contrast, Multilingual BERT was trained on Wikipedia texts, where the Finnish Wikipedia text is approximately 3% of the amount used to train `FinBERT`.

These features allow `FinBERT` to outperform not only Multilingual BERT but also all previously proposed models when fine-tuned for Finnish natural language processing tasks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_finnish_uncased_fi_2.6.0_2.4_1598897885576.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_finnish_uncased_fi_2.6.0_2.4_1598897885576.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_finnish_uncased", "fi") \
      .setInputCols("sentence") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['Vihaan syöpää', 'antibiootit eivät ole kipulääkkeitä']], ["text"]))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_finnish_uncased", "fi")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("Vihaan syöpää","antibiootit eivät ole kipulääkkeitä").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["Vihaan syöpää", "antibiootit eivät ole kipulääkkeitä"]
embeddings_df = nlu.load('fi.embed_sentence.bert.cased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	sentence	                              fi_embed_sentence_bert_uncased_embeddings
		
      Vihaan syöpää 	                        [-0.32807931303977966, -0.18222537636756897, 0...
 	antibiootit eivät ole kipulääkkeitä 	[-0.192955881357193, -0.11151257902383804, 0.7...
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_finnish_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[fi]|
|Dimension:|768|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from https://github.com/TurkuNLP/FinBERT
