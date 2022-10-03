{%- capture title -%}
DeIdentification
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Deidentifies Input Annotations of types DOCUMENT, TOKEN and CHUNK, by either masking or obfuscating the given CHUNKS.

To create a configured DeIdentificationModel, please see the example of DeIdentification.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_api_link -%}
[DeIdentificationModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentificationModel)
{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import * 
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

bert_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("bert_embeddings")

fin_ner = finance.NerModel.pretrained('finner_deid', "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter =  finance.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ORG": "PARTY"}) # Replace "ORG" entity as "PARTY"

ner_finner = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "bert_embeddings"]) \
    .setOutputCol("ner_finner") 
    #.setLabelCasing("upper")

ner_converter_finner = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_finner"]) \
    .setOutputCol("ner_finner_chunk") \
    .setWhiteList(['ROLE']) # Just use "ROLE" entity from this NER

chunk_merge =  finance.ChunkMergeApproach()\
    .setInputCols("ner_finner_chunk", "ner_chunk")\
    .setOutputCol("deid_merged_chunk")

deidentification =  finance.DeIdentification() \
    .setInputCols(["sentence", "token", "deid_merged_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("mask")\
    .setIgnoreRegex(True)

# Pipeline
data = spark.createDataFrame([
    ["Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon"]
]).toDF("text")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      bert_embeddings,
      fin_ner,
      ner_converter,
      ner_finner,
      ner_converter_finner,
      chunk_merge,
      deidentification])

result = nlpPipeline.fit(data).transform(data)
{%- endcapture -%}

{%- capture model_python_legal -%}
from johnsnowlabs import * 
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

legal_ner = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter = legal.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ALIAS": "PARTY"})

ner_signers = legal.NerModel.pretrained("legner_signers", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_signers") 
    #.setLabelCasing("upper")

ner_converter_signers = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_signers"]) \
    .setOutputCol("ner_signer_chunk")

chunk_merge = legal.ChunkMergeApproach()\
    .setInputCols("ner_signer_chunk", "ner_chunk")\
    .setOutputCol("deid_merged_chunk")

deidentification = legal.DeIdentification() \
    .setInputCols(["sentence", "token", "deid_merged_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("mask")\
    .setIgnoreRegex(True)

# Pipeline
data = spark.createDataFrame([
    ["ENTIRE AGREEMENT.  This Agreement contains the entire understanding of the parties hereto with respect to the transactions and matters contemplated hereby, supersedes all previous Agreements between i-Escrow and 2TheMart concerning the subject matter.

2THEMART.COM, INC.:                         I-ESCROW, INC.:

By:Dominic J. Magliarditi                By:Sanjay Bajaj Name: Dominic J. Magliarditi                Name: Sanjay Bajaj Title: President                            Title: VP Business Development Date: 6/21/99                               Date: 6/11/99 "]
]).toDF("text")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      legal_ner,
      ner_converter,
      ner_signers,
      ner_converter_signers,
      chunk_merge,
      deidentification])

result = nlpPipeline.fit(data).transform(data)
{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new nlp.SentenceDetector()
    .setInputCols(["document"])
    .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
    .setInputCols(["sentence"])
    .setOutputCol("token")

val embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val bert_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("bert_embeddings")

val fin_ner = finance.NerModel.pretrained('finner_deid', "en", "finance/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

val ner_converter =  finance.NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")
    .setReplaceLabels({"ORG": "PARTY"}) # Replace "ORG" entity as "PARTY"

val ner_finner = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")
    .setInputCols(Array("sentence", "token", "bert_embeddings"))
    .setOutputCol("ner_finner") 
    #.setLabelCasing("upper")

val ner_converter_finner = new nlp.NerConverter()
    .setInputCols(Array("sentence", "token", "ner_finner"))
    .setOutputCol("ner_finner_chunk")
    .setWhiteList(['ROLE']) # Just use "ROLE" entity from this NER

val chunk_merge =  new finance.ChunkMergeApproach()
    .setInputCols(Array("ner_finner_chunk", "ner_chunk"))
    .setOutputCol("deid_merged_chunk")

val deidentification =  new finance.DeIdentification()
    .setInputCols(Array("sentence", "token", "deid_merged_chunk"))
    .setOutputCol("deidentified")
    .setMode("mask")
    .setIgnoreRegex(True)

# Pipeline
val data = Seq("Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon").toDF("text")

val nlpPipeline = new Pipeline().setStages(Array(
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      bert_embeddings,
      fin_ner,
      ner_converter,
      ner_finner,
      ner_converter_finner,
      chunk_merge,
      deidentification))

val result = nlpPipeline.fit(data).transform(data)
{%- endcapture -%}

{%- capture model_scala_legal -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new nlp.SentenceDetector()
    .setInputCols(["document"])
    .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
    .setInputCols(["sentence"])
    .setOutputCol("token")

val embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val legal_ner = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

val ner_converter = new legal.NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ALIAS": "PARTY"})

val ner_signers = legal.NerModel.pretrained("legner_signers", "en", "legal/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner_signers") 
    #.setLabelCasing("upper")

val ner_converter_signers = new nlp.NerConverter()
    .setInputCols(Array("sentence", "token", "ner_signers"))
    .setOutputCol("ner_signer_chunk")

val chunk_merge = new legal.ChunkMergeApproach()
    .setInputCols(Array("ner_signer_chunk", "ner_chunk"))
    .setOutputCol("deid_merged_chunk")

val deidentification = new legal.DeIdentification()
    .setInputCols(Array("sentence", "token", "deid_merged_chunk"))
    .setOutputCol("deidentified") \
    .setMode("mask")\
    .setIgnoreRegex(True)

# Pipeline
val data = Seq("ENTIRE AGREEMENT.  This Agreement contains the entire understanding of the parties hereto with respect to the transactions and matters contemplated hereby, supersedes all previous Agreements between i-Escrow and 2TheMart concerning the subject matter.

2THEMART.COM, INC.:                         I-ESCROW, INC.:

By:Dominic J. Magliarditi                By:Sanjay Bajaj Name: Dominic J. Magliarditi                Name: Sanjay Bajaj Title: President                            Title: VP Business Development Date: 6/21/99                               Date: 6/11/99 ").toDF("text")

val nlpPipeline = new Pipeline().setStages(Array(
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      legal_ner,
      ner_converter,
      ner_signers,
      ner_converter_signers,
      chunk_merge,
      deidentification))

val result = nlpPipeline.fit(data).transform(data)
{%- endcapture -%}


{%- capture approach_description -%}
Contains all the methods for training a DeIdentificationModel model.
This module can obfuscate or mask the entities that contains personal information. These can be set with a file of
regex patterns with setRegexPatternsDictionary, where each line is a mapping of
entity to regex.
```
DATE \d{4}
AID \d{6,7}
```

Additionally, obfuscation strings can be defined with setObfuscateRefFile, where each line
is a mapping of string to entity. The format and seperator can be speficied with
setRefFileFormat and setRefSep.
```
Dr. Gregory House#DOCTOR
01010101#MEDICALRECORD
```

Ideally this annotator works in conjunction with Demographic Named EntityRecognizers that can be trained either using
[TextMatchers](/docs/en/annotators#textmatcher),
[RegexMatchers](/docs/en/annotators#regexmatcher),
[DateMatchers](/docs/en/annotators#datematcher),
[NerCRFs](/docs/en/annotators#nercrf) or
[NerDLs](/docs/en/annotators#nerdl)
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN, CHUNK
{%- endcapture -%}

{%- capture approach_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *

documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

 sentenceDetector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Ner entities
clinical_sensitive_entities = medical.NerModel \
    .pretrained("ner_deid_enriched", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")

nerConverter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_con")

# Deidentification
deIdentification = medical.DeIdentification() \
    .setInputCols(["ner_chunk", "token", "sentence"]) \
    .setOutputCol("dei") \
    # file with custom regex pattern for custom entities
    .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt") \
    # file with custom obfuscator names for the entities
    .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt") \
    .setRefFileFormat("csv") \
    .setRefSep("#") \
    .setMode("obfuscate") \
    .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd")) \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setDays(5) \
    .setObfuscateRefSource("file")

# Pipeline
data = spark.createDataFrame([
    ["# 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09."]
]).toDF("text")

pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    clinical_sensitive_entities,
    nerConverter,
    deIdentification
])
result = pipeline.fit(data).transform(data)

# Show Results
result.select("dei.result").show(truncate = False)
+--------------------------------------------------------------------------------------------------+
|result                                                                                            |
+--------------------------------------------------------------------------------------------------+
|[# 01010101 Date : 01/18/93 PCP : Dr. Gregory House , <AGE> years-old , Record date : 2079-11-14.]|
+--------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import *

documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

 sentenceDetector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Ner entities
clinical_sensitive_entities = medical.NerModel \
    .pretrained("ner_deid_enriched", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")

nerConverter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_con")

# Deidentification
deIdentification = legal.DeIdentification() \
    .setInputCols(["ner_chunk", "token", "sentence"]) \
    .setOutputCol("dei") \
    # file with custom regex pattern for custom entities
    .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt") \
    # file with custom obfuscator names for the entities
    .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt") \
    .setRefFileFormat("csv") \
    .setRefSep("#") \
    .setMode("obfuscate") \
    .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd")) \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setDays(5) \
    .setObfuscateRefSource("file")

# Pipeline
pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    clinical_sensitive_entities,
    nerConverter,
    deIdentification
])
{%- endcapture -%}

{%- capture approach_python_finance -%}
from johnsnowlabs import *

documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

 sentenceDetector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Ner entities
clinical_sensitive_entities = medical.NerModel \
    .pretrained("ner_deid_enriched", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")

nerConverter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_con")

# Deidentification
deIdentification = finance.DeIdentification() \
    .setInputCols(["ner_chunk", "token", "sentence"]) \
    .setOutputCol("dei") \
    # file with custom regex pattern for custom entities
    .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt") \
    # file with custom obfuscator names for the entities
    .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt") \
    .setRefFileFormat("csv") \
    .setRefSep("#") \
    .setMode("obfuscate") \
    .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd")) \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setDays(5) \
    .setObfuscateRefSource("file")

# Pipeline
pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    clinical_sensitive_entities,
    nerConverter,
    deIdentification
])
{%- endcapture -%}


{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

 val sentenceDetector = new nlp.SentenceDetector()
     .setInputCols(Array("document"))
     .setOutputCol("sentence")
     .setUseAbbreviations(true)

 val tokenizer = new nlp.Tokenizer()
     .setInputCols(Array("sentence"))
     .setOutputCol("token")

 val embeddings = nlp.WordEmbeddingsModel
     .pretrained("embeddings_clinical", "en", "clinical/models")
     .setInputCols(Array("sentence", "token"))
     .setOutputCol("embeddings")

// Ner entities
val clinical_sensitive_entities = medical.NerModel.pretrained("ner_deid_enriched", "en", "clinical/models")
        .setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

 val nerConverter = new nlp.NerConverter()
     .setInputCols(Array("sentence", "token", "ner"))
     .setOutputCol("ner_con")

// Deidentification
val deIdentification = new medical.DeIdentification()
     .setInputCols(Array("ner_chunk", "token", "sentence"))
     .setOutputCol("dei")
     // file with custom regex patterns for custom entities
     .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt")
     // file with custom obfuscator names for the entities
     .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt")
     .setRefFileFormat("csv")
     .setRefSep("#")
     .setMode("obfuscate")
     .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd"))
     .setObfuscateDate(true)
     .setDateTag("DATE")
     .setDays(5)
     .setObfuscateRefSource("file")

// Pipeline
val data = Seq(
  "# 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09."
).toDF("text")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  clinical_sensitive_entities,
  nerConverter,
  deIdentification
))
val result = pipeline.fit(data).transform(data)

result.select("dei.result").show(truncate = false)

// Show Results
//
// result.select("dei.result").show(truncate = false)
// +--------------------------------------------------------------------------------------------------+
// |result                                                                                            |
// +--------------------------------------------------------------------------------------------------+
// |[# 01010101 Date : 01/18/93 PCP : Dr. Gregory House , <AGE> years-old , Record date : 2079-11-14.]|
// +--------------------------------------------------------------------------------------------------+
//
{%- endcapture -%}

{%- capture approach_scala_legal -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

 val sentenceDetector = new nlp.SentenceDetector()
     .setInputCols(Array("document"))
     .setOutputCol("sentence")
     .setUseAbbreviations(true)

 val tokenizer = new nlp.Tokenizer()
     .setInputCols(Array("sentence"))
     .setOutputCol("token")

 val embeddings = nlp.WordEmbeddingsModel
     .pretrained("embeddings_clinical", "en", "clinical/models")
     .setInputCols(Array("sentence", "token"))
     .setOutputCol("embeddings")

// Ner entities
val clinical_sensitive_entities = medical.NerModel.pretrained("ner_deid_enriched", "en", "clinical/models")
        .setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

 val nerConverter = new nlp.NerConverter()
     .setInputCols(Array("sentence", "token", "ner"))
     .setOutputCol("ner_con")

// Deidentification
val deIdentification = new legal.DeIdentification()
     .setInputCols(Array("ner_chunk", "token", "sentence"))
     .setOutputCol("dei")
     // file with custom regex patterns for custom entities
     .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt")
     // file with custom obfuscator names for the entities
     .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt")
     .setRefFileFormat("csv")
     .setRefSep("#")
     .setMode("obfuscate")
     .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd"))
     .setObfuscateDate(true)
     .setDateTag("DATE")
     .setDays(5)
     .setObfuscateRefSource("file")

// Pipeline

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  clinical_sensitive_entities,
  nerConverter,
  deIdentification
))
{%- endcapture -%}

{%- capture approach_scala_finance -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

 val sentenceDetector = new nlp.SentenceDetector()
     .setInputCols(Array("document"))
     .setOutputCol("sentence")
     .setUseAbbreviations(true)

 val tokenizer = new nlp.Tokenizer()
     .setInputCols(Array("sentence"))
     .setOutputCol("token")

 val embeddings = nlp.WordEmbeddingsModel
     .pretrained("embeddings_clinical", "en", "clinical/models")
     .setInputCols(Array("sentence", "token"))
     .setOutputCol("embeddings")

// Ner entities
val clinical_sensitive_entities = medical.NerModel.pretrained("ner_deid_enriched", "en", "clinical/models")
        .setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

 val nerConverter = new nlp.NerConverter()
     .setInputCols(Array("sentence", "token", "ner"))
     .setOutputCol("ner_con")

// Deidentification
val deIdentification = new finance.DeIdentification()
     .setInputCols(Array("ner_chunk", "token", "sentence"))
     .setOutputCol("dei")
     // file with custom regex patterns for custom entities
     .setRegexPatternsDictionary("path/to/dic_regex_patterns_main_categories.txt")
     // file with custom obfuscator names for the entities
     .setObfuscateRefFile("path/to/obfuscate_fixed_entities.txt")
     .setRefFileFormat("csv")
     .setRefSep("#")
     .setMode("obfuscate")
     .setDateFormats(Array("MM/dd/yy","yyyy-MM-dd"))
     .setObfuscateDate(true)
     .setDateTag("DATE")
     .setDays(5)
     .setObfuscateRefSource("file")

// Pipeline

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  clinical_sensitive_entities,
  nerConverter,
  deIdentification
))
{%- endcapture -%}

{%- capture approach_api_link -%}
[DeIdentification](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentification)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
model_python_finance=model_python_finance
model_python_legal=model_python_legal
model_scala_finance=model_scala_finance
model_scala_legal=model_scala_legal
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_legal=approach_python_legal
approach_python_finance=approach_python_finance
approach_scala_medical=approach_scala_medical
approach_scala_legal=approach_scala_legal
approach_scala_finance=approach_scala_finance
approach_api_link=approach_api_link
%}
