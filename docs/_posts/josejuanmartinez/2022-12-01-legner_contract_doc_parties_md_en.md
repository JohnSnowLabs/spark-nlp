---
layout: model
title: Legal NER (Parties, Dates, Document Type - md)
author: John Snow Labs
name: legner_contract_doc_parties_md
date: 2022-12-01
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 

This is a Legal NER Model, aimed to process the first page of the agreements when information can be found about:
- Parties of the contract/agreement;
- Aliases of those parties, or how those parties will be called further on in the document;
- Document Type;
- Effective Date of the agreement;

This model can be used all along with its Relation Extraction model to retrieve the relations between these entities, called `legre_contract_doc_parties`

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives outside the specific section, which may help to run it at whole document level (although not recommended).

## Predicted Entities

`PARTY`, `EFFDATE`, `DOC`, `ALIAS`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_PARTIES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_md_en_1.0.0_3.0_1669892999925.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_md_en_1.0.0_3.0_1669892999925.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")\

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties_md', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""
INTELLECTUAL PROPERTY AGREEMENT

This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties").
"""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+------------+---------+
|       token|ner_label|
+------------+---------+
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|        This|        O|
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|           (|        O|
|        this|        O|
|           "|        O|
|   Agreement|        O|
|         "),|        O|
|       dated|        O|
|          as|        O|
|          of|        O|
|    December|B-EFFDATE|
|          31|I-EFFDATE|
|           ,|I-EFFDATE|
|        2018|I-EFFDATE|
|           (|        O|
|         the|        O|
|           "|        O|
|   Effective|        O|
|        Date|        O|
|          ")|        O|
|          is|        O|
|     entered|        O|
|        into|        O|
|          by|        O|
|         and|        O|
|     between|        O|
|   Armstrong|  B-PARTY|
|    Flooring|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|          .,|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|      Seller|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AFI|  B-PARTY|
|   Licensing|  I-PARTY|
|         LLC|  I-PARTY|
|           ,|        O|
|           a|        O|
|    Delaware|        O|
|     limited|        O|
|   liability|        O|
|     company|        O|
|          ("|        O|
|   Licensing|  B-ALIAS|
|           "|        O|
|         and|        O|
|    together|        O|
|        with|        O|
|      Seller|  B-ALIAS|
|           ,|        O|
|           "|        O|
|     Arizona|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AHF|  B-PARTY|
|     Holding|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|           .|        O|
|           (|        O|
|    formerly|        O|
|       known|        O|
|          as|        O|
|      Tarzan|        O|
|      HoldCo|        O|
|           ,|        O|
|         Inc|        O|
|         .),|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|       Buyer|  B-ALIAS|
|          ")|        O|
|         and|        O|
|   Armstrong|  B-PARTY|
|    Hardwood|  I-PARTY|
|    Flooring|  I-PARTY|
|     Company|  I-PARTY|
------------------------
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_contract_doc_parties_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-PARTY	 413	 57	 64	 0.8787234	 0.8658281	 0.8722281
B-EFFDATE	 43	 2	 5	 0.95555556	 0.8958333	 0.92473114
B-DOC	 75	 7	 16	 0.91463417	 0.82417583	 0.867052
I-EFFDATE	 138	 6	 8	 0.9583333	 0.94520545	 0.9517241
I-ALIAS	 5	 0	 4	 1.0	 0.5555556	 0.71428573
I-DOC	 176	 20	 40	 0.8979592	 0.8148148	 0.8543689
I-FORMER_PARTY_NAME	 2	 0	 0	 1.0	 1.0	 1.0
B-PARTY	 141	 21	 34	 0.8703704	 0.8057143	 0.8367952
B-FORMER_PARTY_NAME	 1	 0	 0	 1.0	 1.0	 1.0
B-ALIAS	 66	 4	 11	 0.94285715	 0.85714287	 0.89795923
Macro-average	 66 4 11 0.94184333 0.856427 0.8971066
Micro-average	 66 4 11 0.9005947 0.85346216 0.87639517
```