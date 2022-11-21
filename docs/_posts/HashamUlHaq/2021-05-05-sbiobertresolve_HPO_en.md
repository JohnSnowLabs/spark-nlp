---
layout: model
title: Entity Resolver for Human Phenotype Ontology
author: John Snow Labs
name: sbiobertresolve_HPO
date: 2021-05-05
tags: [en, licensed, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps phenotypic abnormalities encountered in human diseases to Human Phenotype Ontology (HPO) codes.

## Predicted Entities

This model returns Human Phenotype Ontology (HPO) codes for phenotypic abnormalities encountered in human diseases. It also returns associated codes from the following vocabularies for each HPO code: 

- MeSH (Medical Subject Headings)
- SNOMED
- UMLS (Unified Medical Language System ) 
- ORPHA (international reference resource for information on rare diseases and orphan drugs) 
- OMIM (Online Mendelian Inheritance in Man)

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_HPO/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_HPO_en_3.0.2_3.0_1620235451661.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

```sbiobertresolve_HPO``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_human_phenotype_gene_clinical``` as NER model. No need to ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_HPO", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokens, embeddings, ner, ner_converter, chunk2doc, sbert_embedder, resolver])

model = LightPipeline(pipeline.fit(spark.createDataFrame([['']], ["text"])))

text="""These disorders include cancer, bipolar disorder, schizophrenia, autism, Cri-du-chat syndrome, myopia, cortical cataract-linked Alzheimer's disease, and infectious diseases"""

results = model.fullAnnotate(text)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.HPO").predict("""These disorders include cancer, bipolar disorder, schizophrenia, autism, Cri-du-chat syndrome, myopia, cortical cataract-linked Alzheimer's disease, and infectious diseases""")
```

</div>

## Results

```bash
|    | chunk            | entity   | resolution   | aux_codes                                                                    |
|---:|:-----------------|:---------|:-------------|:-----------------------------------------------------------------------------|
|  0 | cancer           | HP       | HP:0002664   | MSH:D009369||SNOMED:108369006,363346000||UMLS:C0006826,C0027651||ORPHA:1775  |
|  1 | bipolar disorder | HP       | HP:0007302   | MSH:D001714||SNOMED:13746004||UMLS:C0005586||ORPHA:370079                    |
|  2 | schizophrenia    | HP       | HP:0100753   | MSH:D012559||SNOMED:191526005,58214004||UMLS:C0036341||ORPHA:231169          |
|  3 | autism           | HP       | HP:0000717   | MSH:D001321||SNOMED:408856003,408857007,43614003||UMLS:C0004352||ORPHA:79279 |
|  4 | myopia           | HP       | HP:0000545   | MSH:D009216||SNOMED:57190000||UMLS:C0027092||ORPHA:370022                    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_HPO|
|Compatibility:|Healthcare NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[hpo_code]|
|Language:|en|
