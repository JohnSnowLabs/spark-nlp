{%- capture title -%}
LongformerEmbeddings
{%- endcapture -%}

{%- capture description -%}
Longformer is a transformer model for long documents. The Longformer model was presented in [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
longformer-base-4096 is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents.
It supports sequences of length up to 4,096.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = LongformerEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
```
The default model is `"longformer_base_4096"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For some examples of usage, see [LongformerEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddingsTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are compatible with Spark NLP 🚀. The Spark NLP Workshop
example shows how to import them https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.

**Paper Abstract:**

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length.
To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer.
Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention.
Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8.
In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks.
Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.
We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.*

The original code can be found ```here``` https://github.com/allenai/longformer.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = LongformerEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
|[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
|[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
|[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
|[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = LongformerEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
|[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
|[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
|[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
|[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[LongformerEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[LongformerEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
source_link=source_link
%}