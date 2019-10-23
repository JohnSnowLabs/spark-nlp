---
layout: article
title: Licensed Models
permalink: /docs/en/licensed_models
key: docs-licensed-models
modify_date: "2019-10-23"
---

## Pretrained Models

`pretrained(name, lang)` function to use

### English - Licensed Enterprise

It is required to specify 3rd argument to `pretrained(name, lang, loc)` function (location) to add the location of these 

| Model                                  |   name     |   language     |   loc     |
|----------------------------------------|---------------|---------------|---------------|
|NerDLModel        |`ner_clinical`|en|clinical/models|
|AssertionLogRegModel        |`assertion_ml`|en|clinical/models|
|AssertionDLModel        |`assertion_dl`|en|clinical/models|
|NerDLModel        |`deidentify_dl`|en|clinical/models|
|DeIdentificationModel        |`deidentify_rb`|en|clinical/models|
|WordEmbeddingsModel        |`embeddings_clinical`|en|clinical/models|
|PerceptronModel        |`pos_clinical`|en|clinical/models|
|EntityResolverModel        |`resolve_icd10`|en|clinical/models|
|EntityResolverModel        |`resolve_icd10cm_cl_em`|en|clinical/models|
|EntityResolverModel        |`resolve_icd10pcs_cl_em`|en|clinical/models|
|ContextSpellCheckerModel        |`context_spell_med`|en|clinical/models|


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
