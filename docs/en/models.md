---
layout: article
title: Models
permalink: /docs/en/models
key: docs-models
modify_date: "2020-09-10"
---

## Pretrained Models

Pretrained Models moved to its own dedicated repository.
Please follow this link for the updated list:
[https://github.com/JohnSnowLabs/spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models)
{:.success}

## How to use Pretrained Models

### Online

You can follow this approach to use Spark NLP pretrained models:

```python
# load NER model trained by deep learning approach and GloVe word embeddings
ner_dl = NerDLModel.pretrained('ner_dl')
# load NER model trained by deep learning approach and BERT word embeddings
ner_bert = NerDLModel.pretrained('ner_dl_bert')
```

The default language is `en`, so for other laguages you should set the language:

```scala
// load French POS tagger model trained by Universal Dependencies
val french_pos = PerceptronModel.pretrained("pos_ud_gsd", lang="fr")
// load Italain LemmatizerModel
val italian_lemma = LemmatizerModel.pretrained("lemma_dxc", lang="it")
````

### Offline

If you have any trouble using online pipelines or models in your environment (maybe it's air-gapped), you can directly download them for `offline` use.

After downloading offline models/pipelines and extracting them, here is how you can use them iside your code (the path could be a shared storage like HDFS in a cluster):

* Loading `PerceptronModel` annotator model inside Spark NLP Pipeline

```scala
val french_pos = PerceptronModel.load("/tmp/pos_ud_gsd_fr_2.0.2_2.4_1556531457346/")
      .setInputCols("document", "token")
      .setOutputCol("pos")
```

## Public Models

If you wish to use a pre-trained model for a specific annotator in your pipeline, you need to use the annotator which is mentioned under `Model` following with `pretrained(name, lang)` function.

Example to load a pretraiand BERT model or NER model:

```python
bert = BertEmbeddings.pretrained(name='bert_base_cased', lang='en')

ner_onto = NerDLModel.pretrained(name='ner_dl_bert', lang='en')
```

Please follow this link for the updated list:
[https://github.com/JohnSnowLabs/spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models)
{:.success}
