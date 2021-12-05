{%- capture title -%}
ChunkKeyPhraseExtraction
{%- endcapture -%}

{%- capture description -%}
Chunk KeyPhrase Extraction uses Bert Sentence Embeddings to determine the most relevant key phrases describing a text. 
The input to the model consists of chunk annotations and sentence or document annotation. The model compares the chunks 
against the corresponding sentences/documents and selects the chunks which are most representative of the broader text 
context (i.e. the document or the sentence they belong to). The key phrases candidates (i.e. the input chunks) can be 
generated in various ways, e.g. by NGramGenerator, TextMatcher or NerConverter. The model operates either at sentence 
(selecting the most descriptive chunks from the sentence they belong to) or at document level. In the latter case, the 
key phrases are selected to represent all the input document annotations.

This model is a subclass of [[BertSentenceEmbeddings]] and shares all parameters with it. It can load any pretrained
BertSentenceEmbeddings model. Available models can be found at the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Sentence+Embeddings).


{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline

documenter = sparknlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentencer = sparknlp.annotators.SentenceDetector() \
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens") \

embeddings = sparknlp.annotators.WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["document", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel() \
    .pretrained("ner_jsl_slim", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter()\
    .setInputCols("sentences", "tokens", "ner_tags")\
    .setOutputCol("ner_chunks")

key_phrase_extractor = ChunkKeyPhraseExtraction\
    .pretrained()\
    .setTopN(1)\
    .setDocumentLevelProcessing(False)\
    .setDivergence(0.4)\
    .setInputCols(["sentences", "ner_chunks"])\
    .setOutputCol("ner_chunk_key_phrases")

pipeline = sparknlp.base.Pipeline() \
    .setStages([documenter, sentencer, tokenizer, embeddings, ner_tagger, ner_converter, key_phrase_extractor])

data = spark.createDataFrame([["Her Diabetes has become type 2 in the last year with her Diabetes.He complains of swelling in his right forearm."]]).toDF("text")
results = pipeline.fit(data).transform(data)
results\
    .selectExpr("explode(ner_chunk_key_phrases) AS key_phrase")\
    .selectExpr(
        "key_phrase.result",
        "key_phrase.metadata.entity",
        "key_phrase.metadata.DocumentSimilarity",
        "key_phrase.metadata.MMRScore")\
    .show(truncate=False)
    
+-----------------------------+------------------+-------------------+
|result                       |DocumentSimilarity|MMRScore           |
+-----------------------------+------------------+-------------------+
|gestational diabetes mellitus|0.7391447825527298|0.44348688715422274|
|28-year-old                  |0.4366776288430703|0.13577881610104517|
|type two diabetes mellitus   |0.7323921930094919|0.085800103824974  |
+-----------------------------+------------------+-------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("tokens")

val stopWordsCleaner = StopWordsCleaner.pretrained()
    .setInputCols("tokens")
    .setOutputCol("clean_tokens")
    .setCaseSensitive(false)

val nGrams = new NGramGenerator()
    .setInputCols(Array("clean_tokens"))
    .setOutputCol("ngrams")
    .setN(3)


val chunkKeyPhraseExtractor = ChunkKeyPhraseExtraction
    .pretrained()
    .setTopN(2)
    .setDivergence(0.7f)
    .setInputCols(Array("document", "ngrams"))
    .setOutputCol("key_phrases")

val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    tokenizer,
    stopWordsCleaner,
    nGrams,
    chunkKeyPhraseExtractor))

val sampleText = "Her Diabetes has become type 2 in the last year with her Diabetes." +
    " He complains of swelling in his right forearm."

val testDataset = Seq("").toDS.toDF("text")
val result = pipeline.fit(emptyDataset).transform(testDataset)

result
    .selectExpr("explode(key_phrases) AS key_phrase")
    .selectExpr(
        "key_phrase.result",
        "key_phrase.metadata.DocumentSimilarity",
        "key_phrase.metadata.MMRScore")
    .show(truncate=false)

+--------------------------+-------------------+------------------+
|result                    |DocumentSimilarity |MMRScore          |
+--------------------------+-------------------+------------------+
|complains swelling forearm|0.6325718954229369 |0.1897715761677257|
|type 2 year               |0.40181028931546364|-0.189501077108947|
+--------------------------+-------------------+------------------+
 
{%- endcapture -%}

{%- capture api_link -%}
[ChunkKeyPhraseExtraction](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkKeyPhraseExtraction)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}
