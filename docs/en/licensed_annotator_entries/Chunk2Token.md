{%- capture title -%}
Chunk2Token
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
A feature transformer that converts the input array of strings (annotatorType CHUNK) into an
array of chunk-based tokens (annotatorType TOKEN).

When the input is empty, an empty array is returned.

This Annotator is specially convenient when using NGramGenerator annotations as inputs to WordEmbeddingsModels
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Define a pipeline for generating n-grams
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
token = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
ngrammer = nlp.NGramGenerator() \
 .setN(2) \
 .setEnableCumulative(False) \
 .setInputCols(["token"]) \
 .setOutputCol("ngrams") \
 .setDelimiter("_")

# Stage to convert n-gram CHUNKS to TOKEN type
chunk2Token = medical.Chunk2Token().setInputCols(["ngrams"]).setOutputCol("ngram_tokens")
trainingPipeline = Pipeline(stages=[document, sentenceDetector, token, ngrammer, chunk2Token]).fit(data)

result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, False)
    +----------------------------------------------------------------+
    |col                                                             |
    +----------------------------------------------------------------+
    |{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
    |{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
    |{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
    |{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
    |{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
    +----------------------------------------------------------------+

{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 
# Define a pipeline for generating n-grams
document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
token = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
ngrammer = nlp.NGramGenerator() \
 .setN(2) \
 .setEnableCumulative(False) \
 .setInputCols(["token"]) \
 .setOutputCol("ngrams") \
 .setDelimiter("_")

# Stage to convert n-gram CHUNKS to TOKEN type
chunk2Token = legal.Chunk2Token().setInputCols(["ngrams"]).setOutputCol("ngram_tokens")
trainingPipeline = Pipeline(stages=[document, sentenceDetector, token, ngrammer, chunk2Token])
{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import * 
# Define a pipeline for generating n-grams
document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
token = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
ngrammer = nlp.NGramGenerator() \
 .setN(2) \
 .setEnableCumulative(False) \
 .setInputCols(["token"]) \
 .setOutputCol("ngrams") \
 .setDelimiter("_")

# Stage to convert n-gram CHUNKS to TOKEN type
chunk2Token = finance.Chunk2Token().setInputCols(["ngrams"]).setOutputCol("ngram_tokens")
trainingPipeline = Pipeline(stages=[document, sentenceDetector, token, ngrammer, chunk2Token])
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Define a pipeline for generating n-grams
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val token = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")
val ngrammer = new nlp.NGramGenerator()
 .setN(2)
 .setEnableCumulative(false)
 .setInputCols("token")
 .setOutputCol("ngrams")
 .setDelimiter("_")

// Stage to convert n-gram CHUNKS to TOKEN type
val chunk2Token = new medical.Chunk2Token().setInputCols("ngrams").setOutputCol("ngram_tokens")
val trainingPipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, ngrammer, chunk2Token)).fit(data)

val result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, false)
+----------------------------------------------------------------+
|col                                                             |
+----------------------------------------------------------------+
|{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
|{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
|{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
|{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
|{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
+----------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_legal -%}
from johnsnowlabs import * 
// Define a pipeline for generating n-grams

val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val token = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")
val ngrammer = new nlp.NGramGenerator()
 .setN(2)
 .setEnableCumulative(false)
 .setInputCols("token")
 .setOutputCol("ngrams")
 .setDelimiter("_")

// Stage to convert n-gram CHUNKS to TOKEN type
val chunk2Token = new legal.Chunk2Token().setInputCols("ngrams").setOutputCol("ngram_tokens")
val trainingPipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, ngrammer, chunk2Token))
{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 
// Define a pipeline for generating n-grams

val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val token = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")
val ngrammer = new nlp.NGramGenerator()
 .setN(2)
 .setEnableCumulative(false)
 .setInputCols("token")
 .setOutputCol("ngrams")
 .setDelimiter("_")

// Stage to convert n-gram CHUNKS to TOKEN type
val chunk2Token = new finance.Chunk2Token().setInputCols("ngrams").setOutputCol("ngram_tokens")
val trainingPipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, ngrammer, chunk2Token))
{%- endcapture -%}

{%- capture model_api_link -%}
[Chunk2Token](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/Chunk2Token)
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