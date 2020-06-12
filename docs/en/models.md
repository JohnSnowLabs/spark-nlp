---
layout: article
title: Models
permalink: /docs/en/models
key: docs-models
modify_date: "2020-06-12"
---

## Pretrained Models

Pretrained Models moved to its own dedicated repository.
Please follow this link for updated list:
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

**NOTE:** `build` means the model can be downloaded or loaded for that specific version or above. For instance, `2.4.0` can be used in all the releases after `2.4.x` but not before.

### Dutch - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `nl`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_nl_2.5.0_2.4_1588532720582.zip) |
| PerceptronModel (POS UD)     | `pos_ud_alpino`       | 2.5.0 |   `nl`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_alpino_nl_2.5.0_2.4_1588545949009.zip) |
| NerDLModel (glove_100d)  | `wikiner_6B_100`       | 2.5.0 |   `nl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_100_nl_2.5.0_2.4_1588546201140.zip) |
| NerDLModel (glove_6B_300)  | `wikiner_6B_300`     | 2.5.0 |   `nl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_nl_2.5.0_2.4_1588546201483.zip) |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.5.0 |   `nl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_nl_2.5.0_2.4_1588546201484.zip) |

### English - Models

| Model                                    | Name                      | Build            | Lang | Offline
|:-----------------------------------------|:--------------------------|:-----------------|:------------|:------|
| LemmatizerModel (Lemmatizer)             | `lemma_antbnc`            | 2.0.2 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_antbnc_en_2.0.2_2.4_1556480454569.zip) |
| PerceptronModel (POS)                    | `pos_anc`                 | 2.0.2 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_anc_en_2.0.2_2.4_1556659930154.zip) |
| PerceptronModel (POS UD)                    | `pos_ud_ewt`          | 2.2.2 |       `en`       | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_ewt_en_2.2.2_2.4_1570464827452.zip) |
| NerCrfModel (NER with GloVe)             | `ner_crf`                 | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_crf_en_2.4.0_2.4_1580237286004.zip) |
| NerDLModel (NER with GloVe)              | `ner_dl`                  | 2.4.3 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_en_2.4.3_2.4_1584624950746.zip) |
| NerDLModel (NER with BERT)              | `ner_dl_bert`              | 2.4.3 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_en_2.4.3_2.4_1584624951079.zip) |
| NerDLModel (OntoNotes with GloVe 100d)   | `onto_100`                | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_100_en_2.4.0_2.4_1579729071672.zip) |
| NerDLModel (OntoNotes with GloVe 300d)   | `onto_300`                | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_300_en_2.4.0_2.4_1579729071854.zip) |
| DeepSentenceDetector                     | `ner_dl_sentence`         | 2.4.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_sentence_en_2.4.0_2.4_1580252313303.zip)|
| SymmetricDeleteModel (Spell Checker)     | `spellcheck_sd`           | 2.0.2 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_sd_en_2.0.2_2.4_1556604489934.zip)|
| NorvigSweetingModel (Spell Checker)      | `spellcheck_norvig`       | 2.0.2 |      `en`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1556605026653.zip)|
| ViveknSentimentModel (Sentiment)         | `sentiment_vivekn`        | 2.0.2 |      `en`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip)|
| DependencyParser (Dependency)            | `dependency_conllu`       | 2.0.8 |      `en`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_2.0.8_2.4_1561435004077.zip)|
| TypedDependencyParser (Dependency)       | `dependency_typed_conllu` | 2.0.8 |      `en`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_typed_conllu_en_2.0.8_2.4_1561473259215.zip) |

#### Embeddings

| Model    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| WordEmbeddings (GloVe)            | `glove_100d`              | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_100d_en_2.4.0_2.4_1579690104032.zip) |
| BertEmbeddings                    | `bert_base_uncased`       | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_en_2.4.0_2.4_1580579889322.zip) |
| BertEmbeddings                    | `bert_base_cased`         | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_cased_en_2.4.0_2.4_1580579557778.zip) |
| BertEmbeddings                    | `bert_large_uncased`      | 2.4.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_uncased_en_2.4.0_2.4_1580581306683.zip) |
| BertEmbeddings                    | `bert_large_cased`        | 2.4.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_cased_en_2.4.0_2.4_1580580251298.zip) |
| ElmoEmbeddings                    | `elmo`                    | 2.4.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/elmo_en_2.4.0_2.4_1580488815299.zip)
| UniversalSentenceEncoder  (USE)   | `tfhub_use`              | 2.4.0 |       `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_en_2.4.0_2.4_1587136330099.zip)
| UniversalSentenceEncoder  (USE)   | `tfhub_use_lg`           | 2.4.0 |       `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_lg_en_2.4.0_2.4_1587136993894.zip)
| AlbertEmbeddings                  | `albert_base_uncased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_uncased_en_2.5.0_2.4_1588073363475.zip)
| AlbertEmbeddings                  | `albert_large_uncased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_large_uncased_en_2.5.0_2.4_1588073397355.zip)
| AlbertEmbeddings                  | `albert_xlarge_uncased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xlarge_uncased_en_2.5.0_2.4_1588073443653.zip)
| AlbertEmbeddings                  | `albert_xxlarge_uncased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xxlarge_uncased_en_2.5.0_2.4_1588073588232.zip)
| XlnetEmbeddings                  | `xlnet_base_cased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_base_cased_en_2.5.0_2.4_1588074114942.zip)
| XlnetEmbeddings                  | `xlnet_large_cased`    | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_large_cased_en_2.5.0_2.4_1588074397954.zip)

#### Classification

| Model    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| ClassifierDL (with tfhub_use)          | `classifierdl_use_trec6`        | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec6_en_2.5.0_2.4_1588492648979.zip) |
| ClassifierDL (with tfhub_use)          | `classifierdl_use_trec50`       | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec50_en_2.5.0_2.4_1588493558481.zip) |
| SentimentDL (with tfhub_use)           | `sentimentdl_use_imdb`          | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_use_imdb_en_2.5.0_2.4_1588679956272.zip) |
| SentimentDL (with tfhub_use)           | `sentimentdl_use_twitter`       | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_use_twitter_en_2.5.0_2.4_1589108892106.zip) |
| SentimentDL (with glove_100d)          | `sentimentdl_glove_imdb`         | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_glove_imdb_en_2.5.0_2.4_1588682682507.zip) |

### French - Models

| Model                        | Name               | Build | Lang |  Offline                                                                                                                                                                                                |
|:-----------------------------|:-------------------|:------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            |   2.0.2    |   `fr`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fr_2.0.2_2.4_1556531462843.zip)                                                                                       |
| PerceptronModel (POS UD)     | `pos_ud_gsd`       |   2.0.2    |    `fr`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_fr_2.0.2_2.4_1556531457346.zip)                                                                                  |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` |   2.0.2    |    `fr`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_fr_2.4.0_2.4_1579699913554.zip)                                                                            |

| Feature   | Description                                                                                                                                                                                            |    |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---|
| **Lemma** | Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`                                                                                                                     |    |
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/fr_gsd/index.html)                                                             |    |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |    |

### German - Models

| Model                        | Name               | Build            | Lang | Offline                                                                                                                                                                                                |
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.0.8 |       `de`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_de_2.0.8_2.4_1561248996126.zip)                                                                                             |
| PerceptronModel (POS UD)     | `pos_ud_hdt`       | 2.0.8 |       `de`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_hdt_de_2.0.8_2.4_1561232528570.zip)                                                                                        |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.4.0 |       `de`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_de_2.4.0_2.4_1579699913555.zip)|

| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`                                                                                                                     |
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/de_hdt/index.html)                                                             |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |

### Italian - Models

| Model                         | Name               | Build            | Lang  | Offline|
|:------------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer)  | `lemma_dxc`        | 2.0.2 |    `it`   |  [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_dxc_it_2.0.2_2.4_1556531469058.zip)        |
| ViveknSentimentAnalysis (Sentiment) | `sentiment_dxc`    | 2.0.2 |    `it`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_dxc_it_2.0.2_2.4_1556531477694.zip)    |
| PerceptronModel (POS UD)      | `pos_ud_isdt`      | 2.0.8 |    `it`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_isdt_it_2.0.8_2.4_1560168427464.zip)      |
| NerDLModel (glove_840B_300)   | `wikiner_840B_300` | 2.4.0 |   `it`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_it_2.4.0_2.4_1579699913554.zip) |

| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on **DXC Technology** dataset                                                                                                                                      |
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/it_isdt/index.html)                                                            |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |

### Norwegian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `nb`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_nb_2.5.0_2.4_1588693886432.zip) |
| PerceptronModel (POS UD)     | `pos_ud_nynorsk`       | 2.5.0 |   `nn`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_nynorsk_nn_2.5.0_2.4_1588693690964.zip) |
| PerceptronModel (POS UD)     | `pos_ud_bokmaal`       | 2.5.0 |   `nb`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_bokmaal_nb_2.5.0_2.4_1588693881973.zip) |
| NerDLModel (glove_100d)  | `norne_6B_100`       | 2.5.0 |   `no`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norne_6B_100_no_2.5.0_2.4_1588781289907.zip) |
| NerDLModel (glove_6B_300)  | `norne_6B_300`     | 2.5.0 |   `no`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norne_6B_300_no_2.5.0_2.4_1588781290264.zip) |
| NerDLModel (glove_840B_300)  | `norne_840B_300` | 2.5.0 |   `no`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norne_840B_300_no_2.5.0_2.4_1588781290267.zip) |

### Polish - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `pl`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_pl_2.5.0_2.4_1588518491035.zip) |
| PerceptronModel (POS UD)     | `pos_ud_lfg`       | 2.5.0 |   `pl`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_lfg_pl_2.5.0_2.4_1588518541171.zip) |
| NerDLModel (glove_100d)  | `wikiner_6B_100`       | 2.5.0 |   `pl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_100_pl_2.5.0_2.4_1588519719293.zip) |
| NerDLModel (glove_6B_300)  | `wikiner_6B_300`     | 2.5.0 |   `pl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_pl_2.5.0_2.4_1588519719571.zip) |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.5.0 |   `pl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_pl_2.5.0_2.4_1588519719572.zip) |

### Portuguese - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `pt`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_pt_2.5.0_2.4_1588499301752.zip) |
| PerceptronModel (POS UD)     | `pos_ud_bosque`       | 2.5.0 |   `pt`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_bosque_pt_2.5.0_2.4_1588499443093.zip) |
| NerDLModel (glove_100d)  | `wikiner_6B_100`       | 2.5.0 |   `pt`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_100_pt_2.5.0_2.4_1588495233192.zip) |
| NerDLModel (glove_6B_300)  | `wikiner_6B_300`     | 2.5.0 |   `pt`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_pt_2.5.0_2.4_1588495233641.zip) |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.5.0 |   `pt`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_pt_2.5.0_2.4_1588495233642.zip) |

### Russian - Models

| Model                        | Name               | Build            | Lang  | Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.4.4 |    `ru`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_ru_2.4.4_2.4_1584013425855.zip) |
| PerceptronModel (POS UD)     | `pos_ud_gsd`       | 2.4.4 |    `ru`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_ru_2.4.4_2.4_1584013495069.zip) |
| NerDLModel (glove_100d)  | `wikiner_6B_100` | 2.4.4 |    `ru`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_100_ru_2.4.4_2.4_1584014001452.zip) |
| NerDLModel (glove_6B_300)  | `wikiner_6B_300` | 2.4.4 |  `ru`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_ru_2.4.4_2.4_1584014001694.zip) |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.4.4 |   `ru`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_ru_2.4.4_2.4_1584014001695.zip) |

| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/ru_gsd/index.html)|
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/ru_gsd/index.html)                                                             |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |

### Spanish - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.4.0 |    `es`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_es_2.4.0_2.4_1581890818386.zip) |
| PerceptronModel (POS UD)     | `pos_ud_gsd`       | 2.4.0 |   `es`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_es_2.4.0_2.4_1581891015986.zip) |
| NerDLModel (glove_100d)  | `wikiner_6B_100` | 2.4.0 |    `es`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_100_es_2.4.0_2.4_1581971941700.zip) |
| NerDLModel (glove_6B_300)  | `wikiner_6B_300` | 2.4.0 |  `es`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_es_2.4.0_2.4_1581971942090.zip) |
| NerDLModel (glove_840B_300)  | `wikiner_840B_300` | 2.4.0 |   `es`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_es_2.4.0_2.4_1581971942091.zip;) |

| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`                                                                                                                     |
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/es_gsd/index.html)                                                             |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |

### Bulgarian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `bg`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_bg_2.5.0_2.4_1588666297763.zip) |
| PerceptronModel (POS UD)     | `pos_ud_btb`       | 2.5.0 |   `bg`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_btb_bg_2.5.0_2.4_1588621401140.zip) |

### Czech - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `cs`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_cs_2.5.0_2.4_1588666300042.zip) |
| PerceptronModel (POS UD)     | `pos_ud_pdt`       | 2.5.0 |   `cs`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_pdt_cs_2.5.0_2.4_1588622155494.zip) |

### Greek - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `el`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_el_2.5.0_2.4_1588686951720.zip) |
| PerceptronModel (POS UD)     | `pos_ud_gdt`       | 2.5.0 |   `el`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gdt_el_2.5.0_2.4_1588686949851.zip) |

### Finnish - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `fi`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fi_2.5.0_2.4_1588671290521.zip) |
| PerceptronModel (POS UD)     | `pos_ud_tdt`       | 2.5.0 |   `fi`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_tdt_fi_2.5.0_2.4_1588622348985.zip) |

### Hungarian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `hu`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_hu_2.5.0_2.4_1588671968880.zip) |
| PerceptronModel (POS UD)     | `pos_ud_szeged`       | 2.5.0 |   `hu`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_szeged_hu_2.5.0_2.4_1588671966774.zip) |

### Romanian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `ro`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_ro_2.5.0_2.4_1588666512149.zip) |
| PerceptronModel (POS UD)     | `pos_ud_rrt`       | 2.5.0 |   `ro`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_rrt_ro_2.5.0_2.4_1588622539956.zip) |

### Slovak - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `sk`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_sk_2.5.0_2.4_1588666524270.zip) |
| PerceptronModel (POS UD)     | `pos_ud_snk`       | 2.5.0 |   `sk`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_snk_sk_2.5.0_2.4_1588622627281.zip) |

### Swedish - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `sv`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_sv_2.5.0_2.4_1588666548498.zip) |
| PerceptronModel (POS UD)     | `pos_ud_tal`       | 2.5.0 |   `sv`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_tal_sv_2.5.0_2.4_1588622711284.zip) |

### Turkish - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `tr`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_tr_2.5.0_2.4_1587479962436.zip) |
| PerceptronModel (POS UD)     | `pos_ud_imst`       | 2.5.0 |   `tr`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_imst_tr_2.5.0_2.4_1587480006078.zip) |

### Ukrainian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.0 |   `uk`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_uk_2.5.0_2.4_1588671294202.zip) |
| PerceptronModel (POS UD)     | `pos_ud_iu`       | 2.5.0 |   `uk`    |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_iu_uk_2.5.0_2.4_1588668890963.zip) |

### Multi-language

| Model                        | Name               | Build            | Lang | Offline |
|:-----------------------------|:-------------------|:-----------------|:------|
| WordEmbeddings (GloVe)       | `glove_840B_300`   | 2.4.0 |  `xx`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_840B_300_xx_2.4.0_2.4_1579698926752.zip)   |
| WordEmbeddings (GloVe)       | `glove_6B_300`     | 2.4.0 |   `xx`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_6B_300_xx_2.4.0_2.4_1579698630432.zip)     |
| BertEmbeddings (multi_cased) | `bert_multi_cased` | 2.4.0 |   `xx`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_multi_cased_xx_2.4.0_2.4_1580582335793.zip) |
| LanguageDetectorDL    | `ld_wiki_7`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_7_xx_2.5.0_2.4_1591875673486.zip) |
| LanguageDetectorDL    | `ld_wiki_20`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_20_xx_2.5.0_2.4_1591875680011.zip) |

* The model with 7 languages: Czech, German, English, Spanish, French, Italy, and Slovak
* The model with 20 languages: Bulgarian, Czech, German, Greek, English, Spanish, Finnish, French, Croatian, Hungarian, Italy, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Swedish, Turkish, and Ukrainian
