---
layout: model
title: Crunchbase Company Names Normalization
author: John Snow Labs
name: finel_crunchbase_companynames
date: 2022-08-09
tags: [en, finance, companies, crunchbase, licensed]
task: Entity Resolution
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an Entity Resolution / Entity Linking model, aimed to normalize an input Company Name from an NER component, into the way it's registered in Crunchbase (up to 2015, if existed).

Then, you can use the CrunchBase Chunk Mapper to get information about that company, as for example the Company Sector, Funding, Status, etc.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_crunchbase_companynames_en_1.0.0_3.2_1660041398986.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_crunchbase_companynames_en_1.0.0_3.2_1660041398986.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_crunchbase_companynames", "en", "finance/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("name")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
      stages = [
          documentAssembler,
          embeddings,
          resolver])

lp = LightPipeline(pipelineModel)

lp.fullAnnotate("Shwrm")

```

</div>

## Results

```bash
+--------+---------+-----------------------------------------------------+----------------------------------------------------+----------------------------+
|   chunk|    code |                                            all_codes|                                        resolutions |               all_distances|
+--------+---------+----------------------------------------------------------------------------------------------------------+----------------------------+
|  Shwrm |   Shwrüm| [Shwrüm, Xervmon Inc, ADVANCED CREDIT TECHNOLOGIES] |[Shwrüm, Xervmon Inc, ADVANCED CREDIT TECHNOLOGIES] |  [0.0000, 0.0436, 0.0448,] |
+--------+---------+-----------------------------------------------------+----------------------------------------------------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_crunchbase_companynames|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[original_company_name]|
|Language:|en|
|Size:|119.0 MB|
|Case sensitive:|false|

## References

In-house company permutions based on a dataset available at https://data.world/fiftin/crunchbase-2015
