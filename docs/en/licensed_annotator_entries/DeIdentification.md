{%- capture title -%}
DeIdentification
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

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

 sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Ner entities
clinical_sensitive_entities = MedicalNerModel \
    .pretrained("ner_deid_enriched", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")

nerConverter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_con")

# Deidentification
deIdentification = DeIdentification() \
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

{%- capture approach_scala_example -%}
val documentAssembler = new DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

 val sentenceDetector = new SentenceDetector()
     .setInputCols(Array("document"))
     .setOutputCol("sentence")
     .setUseAbbreviations(true)

 val tokenizer = new Tokenizer()
     .setInputCols(Array("sentence"))
     .setOutputCol("token")

 val embeddings = WordEmbeddingsModel
     .pretrained("embeddings_clinical", "en", "clinical/models")
     .setInputCols(Array("sentence", "token"))
     .setOutputCol("embeddings")

// Ner entities
val clinical_sensitive_entities = MedicalNerModel.pretrained("ner_deid_enriched", "en", "clinical/models")
        .setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

 val nerConverter = new NerConverter()
     .setInputCols(Array("sentence", "token", "ner"))
     .setOutputCol("ner_con")

// Deidentification
val deIdentification = new DeIdentification()
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

{%- capture approach_api_link -%}
[DeIdentification](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentification)
{%- endcapture -%}


{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
