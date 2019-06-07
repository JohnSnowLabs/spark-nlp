---
layout: article
title: Models
permalink: /docs/en/models
key: docs-models
modify_date: "2019-05-29"
---

## Pretrained Models

### English

| Model                                  |   Name     |   en     |
|----------------------------------------|---------------|---------------|
|LemmatizerModel (Lemmatizer)            |  `lemma_antbnc`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_antbnc_en_2.0.2_2.4_1556480454569.zip)
|PerceptronModel (POS)                   |   `pos_anc`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_anc_en_2.0.2_2.4_1556659930154.zip)
|NerCRFModel (NER with GloVe)                       |    `ner_crf`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_crf_en_2.0.2_2.4_1556652790378.zip)
|NerDLModel (NER with GloVe)                        |    `ner_dl`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_en_2.0.2_2.4_1556501490317.zip)
|NerDLModel (NER with BERT)| `ner_dl_bert`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_en_2.0.2_2.4_1556650375261.zip)
|WordEmbeddings (GloVe) | `glove_100d` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_100d_en_2.0.2_2.4_1556534397055.zip)
|WordEmbeddings (GloVe) | `glove_840B_300` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_840B_300_en_2.0.2_2.4_1558645003344.zip)
|WordEmbeddings (GloVe) | `glove_6B_300` |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove_6B_300_en_2.0.2_2.4_1559059806004.zip)
|WordEmbeddings (BERT)  | `bert_uncased` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_uncased_en_2.0.2_2.4_1556651478920.zip)
|DeepSentenceDetector| `ner_dl_sentence`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_sentence_en_2.0.2_2.4_1556666842347.zip)
|ContextSpellCheckerModel (Spell Checker)|   `spellcheck_dl`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_en_2.0.2_2.4_1556479898829.zip)
|SymmetricDeleteModel (Spell Checker)    |   `spellcheck_sd`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_sd_en_2.0.2_2.4_1556604489934.zip)
|NorvigSweetingModel (Spell Checker)     |  `spellcheck_norvig`   | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1556605026653.zip)
|ViveknSentimentModel (Sentiment)        |    `sentiment_vivekn`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip)
|DependencyParser (Dependency)        |    `dependency_conllu`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_2.0.2_2.4_1556649770312.zip)
|TypedDependencyParser (Dependency)        |    `dependency_typed_conllu`    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_typed_conllu_en_2.0.2_2.4_1556656204957.zip)

### Italian

| Model                            | Name      |   it    |
|----------------------------------|-----------|--------------|
|LemmatizerModel (Lemmatizer)      |`lemma_dxc`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_dxc_it_2.0.2_2.4_1556531469058.zip)
|SentimentDetector (Sentiment)     |  `sentiment_dxc`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_dxc_it_2.0.2_2.4_1556531477694.zip)

### French

| Model                         | Name         |   fr    |
|-------------------------------|--------------|--------------|
|PerceptronModel (POS UD)       | `pos_ud_gsd` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_fr_2.0.2_2.4_1556531457346.zip)
|LemmatizerModel (Lemmatizer)| `lemma`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fr_2.0.2_2.4_1556531462843.zip)
|NerDLModel (NER with glove_840B_300)| `ner_dl_lg`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_lg_fr_2.0.2_2.4_1558826556431.zip)
|NerDLModel (NER with glove_6B_300)| `ner_dl_md`|[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_md_fr_2.0.2_2.4_1559117359574.zip)

### Multi-language

|Model                         | Name          |   xx    |
|-------------------------------|--------------|--------------|
|WordEmbeddings (BERT)  | `bert_multi_cased` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_multi_cased_xx_2.0.3_2.4_1557923470812.zip)

## How to use Pretrained Models

### Online

You can follow this approach to use Spark NLP pretrained models:

```python
# load NER model trained by deep learning approach and GloVe word embeddings
ner_dl = NerDLModel.pretrained('ner_dl')
# load NER model trained by deep learning approach and BERT word embeddings
ner_crf = NerDLModel.pretrained('ner_dl_bert')
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

* Loading Offline Pipeline

```scala
val advancedPipeline = PipelineModel.load("/tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/")
// To use the loaded Pipeline for prediction
advancedPipeline.transform(predictionDF)
```