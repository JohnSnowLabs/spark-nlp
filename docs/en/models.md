---
layout: article
title: Models
permalink: /docs/en/models
key: docs-models
modify_date: "2019-07-22"
---

## Pretrained Models

### English

| Model                                  |   Name     |   en     |
|----------------------------------------|---------------|---------------|
|LemmatizerModel (Lemmatizer)            |  `lemma_antbnc`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_antbnc_en_2.0.2_2.4_1556480454569.zip)
|PerceptronModel (POS)                   |   `pos_anc`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_anc_en_2.0.2_2.4_1556659930154.zip)
|NerCRFModel (NER with GloVe)            |    `ner_crf`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_crf_en_2.0.2_2.4_1556652790378.zip)
|NerDLModel (NER with GloVe)             |    `ner_dl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_en_2.0.2_2.4_1558802205173.zip)
|NerDLModel (NER with GloVe)             |    `ner_dl_contrib`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_contrib_en_2.0.2_2.4_1556501490317.zip)
|NerDLModel (NER with BERT)| `ner_dl_bert`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_en_2.0.2_2.4_1558809068913.zip)
|NerDLModel (NER with BERT)| `ner_dl_bert_contrib`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_contrib_en_2.0.2_2.4_1556650375261.zip)
|WordEmbeddings (GloVe) | `glove_100d` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_100d_en_2.0.2_2.4_1556534397055.zip)
|WordEmbeddings (BERT)  | `bert_uncased` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_uncased_en_2.0.2_2.4_1556651478920.zip)
|DeepSentenceDetector| `ner_dl_sentence`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_sentence_en_2.0.2_2.4_1556666842347.zip)
|ContextSpellCheckerModel (Spell Checker)|   `spellcheck_dl`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_en_2.0.2_2.4_1556479898829.zip)
|SymmetricDeleteModel (Spell Checker)    |   `spellcheck_sd`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_sd_en_2.0.2_2.4_1556604489934.zip)
|NorvigSweetingModel (Spell Checker)     |  `spellcheck_norvig`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1556605026653.zip)
|ViveknSentimentModel (Sentiment)        |    `sentiment_vivekn`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip)
|DependencyParser (Dependency)        |    `dependency_conllu`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_2.0.2_2.4_1556649770312.zip)
|TypedDependencyParser (Dependency)        |    `dependency_typed_conllu`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_typed_conllu_en_2.0.2_2.4_1556656204957.zip)

### French

| Model                         | Name         |   fr    |
|-------------------------------|--------------|---------------|
|LemmatizerModel (Lemmatizer)| `lemma`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fr_2.0.2_2.4_1556531462843.zip)
|PerceptronModel (POS UD)       | `pos_ud_gsd` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_fr_2.0.2_2.4_1556531457346.zip)
|NerDLModel (glove_840B_300)| `wikiner_840B_300`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_fr_2.1.0_2.4_1563035043013.zip)

|Feature | Description|
|---|----|
|**Lemma**|Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/fr_gsd/index.html)
|**NER**|Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities

### German

| Model                         | Name         |   de    |
|-------------------------------|--------------|---------------|
|LemmatizerModel (Lemmatizer)    | `lemma`  | [de](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_de_2.0.8_2.4_1561248996126.zip)
|PerceptronModel (POS UD)      |`pos_ud_hdt`| [de](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_hdt_de_2.0.8_2.4_1561232528570.zip)
|NerDLModel (glove_840B_300)| `wikiner_840B_300`|[de](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_de_2.1.0_2.4_1563035544700.zip)

|Feature | Description|
|---|----|
|**Lemma**|Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/de_hdt/index.html)
|**NER**|Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities

### Italian

| Model                            | Name      |   it    |
|----------------------------------|-----------|--------------|
|LemmatizerModel (Lemmatizer)      |`lemma_dxc`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_dxc_it_2.0.2_2.4_1556531469058.zip)
|SentimentDetector (Sentiment)     |  `sentiment_dxc`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_dxc_it_2.0.2_2.4_1556531477694.zip)
|PerceptronModel (POS UD)      |`pos_ud_isdt`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_isdt_it_2.0.8_2.4_1560168427464.zip)
|NerDLModel (glove_840B_300)| `wikiner_840B_300`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_840B_300_it_2.1.0_2.4_1563095099139.zip)

|Feature | Description|
|---|----|
|**Lemma**|Trained by **Lemmatizer** annotator on **DXC Technology** dataset
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/it_isdt/index.html)
|**NER**|Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities

### Multi-language

|Model                         | Name          |   xx    |
|-------------------------------|--------------|--------------|
|WordEmbeddings (GloVe) | `glove_840B_300` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_840B_300_xx_2.0.2_2.4_1558645003344.zip)
|WordEmbeddings (GloVe) | `glove_6B_300` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_6B_300_xx_2.0.2_2.4_1559059806004.zip)|
|WordEmbeddings (BERT)  | `bert_multi_cased` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_multi_cased_xx_2.0.3_2.4_1557923470812.zip)

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
