{%- capture title -%}
NerChunker
{%- endcapture -%}

{%- capture description -%}
Extracts phrases that fits into a known pattern using the NER tags. Useful for entity groups with neighboring tokens
when there is no pretrained NER model to address certain issues. A Regex needs to be provided to extract the tokens
between entities.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, NAMED_ENTITY
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Defining pipeline stages for NER
data= spark.createDataFrame([["She has cystic cyst on her kidney."]]).toDF("text")

documentAssembler= DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector= SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence") \
  .setUseAbbreviations(False)

tokenizer= Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentence","token"]) \
  .setOutputCol("embeddings") \
  .setCaseSensitive(False)

ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") \
  .setInputCols(["sentence","token","embeddings"]) \
  .setOutputCol("ner") \
  .setIncludeConfidence(True)

# Define the NerChunker to combine to chunks
chunker = NerChunker() \
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

{%- capture scala_example -%}
// Defining pipeline stages for NER
val data= Seq("She has cystic cyst on her kidney.").toDF("text")

val documentAssembler=new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector=new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setUseAbbreviations(false)

val tokenizer=new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentence","token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models")
  .setInputCols("sentence","token","embeddings")
  .setOutputCol("ner")
  .setIncludeConfidence(true)

// Define the NerChunker to combine to chunks
val chunker = new NerChunker()
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

{%- capture api_link -%}
[NerChunker](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerChunker)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}