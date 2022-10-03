{%- capture title -%}
NerChunker
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Extracts phrases that fits into a known pattern using the NER tags. Useful for entity groups with neighboring tokens
when there is no pretrained NER model to address certain issues. A Regex needs to be provided to extract the tokens
between entities.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, NAMED_ENTITY
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Defining pipeline stages for NER
data= spark.createDataFrame([["She has cystic cyst on her kidney."]]).toDF("text")

documentAssembler= nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector= nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence") \
  .setUseAbbreviations(False)

tokenizer= nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentence","token"]) \
  .setOutputCol("embeddings") \
  .setCaseSensitive(False)

ner = medical.NerModel.pretrained("ner_radiology", "en", "clinical/models") \
  .setInputCols(["sentence","token","embeddings"]) \
  .setOutputCol("ner") \
  .setIncludeConfidence(True)

# Define the NerChunker to combine to chunks
chunker = medical.NerChunker() \
  .setInputCols(["sentence","ner"]) \
  .setOutputCol("ner_chunk") \
  .setRegexParsers(["<ImagingFindings>.*<BodyPart>"])

pipeline= Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner,
  chunker
])

result = pipeline.fit(data).transform(data)

# Show results:
result.selectExpr("explode(arrays_zip(ner.metadata , ner.result))")
  .selectExpr("col['0'].word as word" , "col['1'] as ner").show(truncate=False)
+------+-----------------+
|word  |ner              |
+------+-----------------+
|She   |O                |
|has   |O                |
|cystic|B-ImagingFindings|
|cyst  |I-ImagingFindings|
|on    |O                |
|her   |O                |
|kidney|B-BodyPart       |
|.     |O                |
+------+-----------------+

result.select("ner_chunk.result").show(truncate=False)
+---------------------------+
|result                     |
+---------------------------+
|[cystic cyst on her kidney]|
+---------------------------+
{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 
# Defining pipeline stages for NER


documentAssembler= nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector= nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence") \
  .setUseAbbreviations(False)

tokenizer= nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentence","token"]) \
  .setOutputCol("embeddings") \
  .setCaseSensitive(False)

ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")

# Define the NerChunker to combine to chunks
chunker = legal.NerChunker() \
  .setInputCols(["sentence","ner"]) \
  .setOutputCol("ner_chunk") \
  .setRegexParsers(["<ImagingFindings>.*<BodyPart>"])

pipeline= Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  chunker
])
{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import * 
# Defining pipeline stages for NER


documentAssembler= nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector= nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence") \
  .setUseAbbreviations(False)

tokenizer= nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentence","token"]) \
  .setOutputCol("embeddings") \
  .setCaseSensitive(False)

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

# Define the NerChunker to combine to chunks
chunker = finance.NerChunker() \
  .setInputCols(["sentence","ner"]) \
  .setOutputCol("ner_chunk") \
  .setRegexParsers(["<ImagingFindings>.*<BodyPart>"])

pipeline= Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  chunker
])
{%- endcapture -%}


{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Defining pipeline stages for NER
val data= Seq("She has cystic cyst on her kidney.").toDF("text")

val documentAssembler=new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector=new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setUseAbbreviations(False)

val tokenizer=new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")
  .setCaseSensitive(False)

val ner = medical.NerModel.pretrained("ner_radiology", "en", "clinical/models")
  .setInputCols(Array("sentence","token","embeddings"))
  .setOutputCol("ner")
  .setIncludeConfidence(True)

// Define the NerChunker to combine to chunks
val chunker = new medical.NerChunker()
  .setInputCols(Array("sentence","ner"))
  .setOutputCol("ner_chunk")
  .setRegexParsers(Array("<ImagingFindings>.<BodyPart>"))

val pipeline=new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner,
  chunker
))

val result = pipeline.fit(data).transform(data)

// Show results:
//
// result.selectExpr("explode(arrays_zip(ner.metadata , ner.result))")
//   .selectExpr("col['0'].word as word" , "col['1'] as ner").show(truncate=false)
// +------+-----------------+
// |word  |ner              |
// +------+-----------------+
// |She   |O                |
// |has   |O                |
// |cystic|B-ImagingFindings|
// |cyst  |I-ImagingFindings|
// |on    |O                |
// |her   |O                |
// |kidney|B-BodyPart       |
// |.     |O                |
// +------+-----------------+
// result.select("ner_chunk.result").show(truncate=false)
// +---------------------------+
// |result                     |
// +---------------------------+
// |[cystic cyst on her kidney]|
// +---------------------------+
//
{%- endcapture -%}


{%- capture model_scala_legal -%}
from johnsnowlabs import * 
// Defining pipeline stages for NER
val documentAssembler=new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector=new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setUseAbbreviations(False)

val tokenizer=new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")
  .setCaseSensitive(False)

val ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")\
  .setInputCols(Array("sentence", "token", "embeddings"))\
  .setOutputCol("ner")

// Define the NerChunker to combine to chunks
val chunker = new legal.NerChunker()
  .setInputCols(Array("sentence","ner"))
  .setOutputCol("ner_chunk")
  .setRegexParsers(Array("<ImagingFindings>.<BodyPart>"))

val pipeline=new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  chunker
))
{%- endcapture -%}


{%- capture model_scala_finance -%}
from johnsnowlabs import * 
// Defining pipeline stages for NER
val documentAssembler=new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector=new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setUseAbbreviations(False)

val tokenizer=new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")
  .setCaseSensitive(False)

val ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
  .setInputCols(Array("sentence", "token", "embeddings")) \
  .setOutputCol("ner")

// Define the NerChunker to combine to chunks
val chunker = new finance.NerChunker()
  .setInputCols(Array("sentence","ner"))
  .setOutputCol("ner_chunk")
  .setRegexParsers(Array("<ImagingFindings>.<BodyPart>"))

val pipeline=new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  chunker
))
{%- endcapture -%}


{%- capture model_api_link -%}
[NerChunker](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerChunker)
{%- endcapture -%}



{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_legal=model_python_legal
model_python_finance=model_python_finance
model_scala_medical=model_scala_medical
model_scala_legal=model_scala_legal
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}