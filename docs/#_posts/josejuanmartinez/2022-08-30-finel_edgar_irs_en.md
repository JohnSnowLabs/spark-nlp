---
layout: model
title: Company Name to IRS (Edgar database)
author: John Snow Labs
name: finel_edgar_irs
date: 2022-08-30
tags: [en, finance, companies, edgar, licensed]
task: Entity Resolution
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an Entity Linking / Entity Resolution model, which allows you to retrieve the IRS number of a company given its name, using SEC Edgar database.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_edgar_irs_en_1.0.0_3.2_1661866402930.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk") \
      .setOutputCol("sentence_embeddings")
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_edgar_irs", "en", "finance/models")\
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("irs_code")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
      stages = [
          documentAssembler,
          embeddings,
          resolver])

lp = LightPipeline(pipelineModel)

lp.fullAnnotate("CONTACT GOLD")
```

</div>

## Results

```bash
+--------------+-----------+---------------------------------------------------------+--------------------------------------------------------+-------------------------------------------+
|         chunk|     code  |                                                all_codes|                                            resolutions |                              all_distances|
+--------------+-----------+---------------------------------------------------------+--------------------------------------------------------+-------------------------------------------+
| CONTACT GOLD |  981369960| [981369960, 271989147, 208531222, 273566922, 270348508] |[981369960, 271989147, 208531222, 273566922, 270348508] |  [0.1733, 0.3700, 0.3867, 0.4103, 0.4121] |
+--------------+-----------+---------------------------------------------------------+--------------------------------------------------------+-------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_edgar_irs|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[company_irs_number]|
|Language:|en|
|Size:|313.8 MB|
|Case sensitive:|false|

## References

In-house scrapping and postprocessing of SEC Edgar Database