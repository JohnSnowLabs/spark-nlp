{%- capture title -%}
BertSentenceEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence-level embeddings using BERT. BERT (Bidirectional Encoder Representations from Transformers) provides dense
vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = BertSentenceEmbeddings.pretrained()
  .setInputCols("sentence")
  .setOutputCol("sentence_bert_embeddings")
```
The default model is `"sent_small_bert_L2_768"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
and the [BertSentenceEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddingsTestSpec.scala).

**Sources** :

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

https://github.com/google-research/bert

**Paper abstract**

*We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a
result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create
state-of-the-art models for a wide range of tasks, such as question answering and language inference, without
substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It
obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score
to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1
question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point
absolute improvement).*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \
    .setInputCols(["sentence"]) \
    .setOutputCol("sentence_bert_embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_bert_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["John loves apples. Mary loves oranges. John loves Mary."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.8951074481010437,0.13753940165042877,0.3108254075050354,-1.65693199634552...|
|[-0.6180210709571838,-0.12179657071828842,-0.191165953874588,-1.4497021436691...|
|[-0.822715163230896,0.7568016648292542,-0.1165061742067337,-1.59048593044281,...|
+--------------------------------------------------------------------------------+

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

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128")
  .setInputCols("sentence")
  .setOutputCol("sentence_bert_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("sentence_bert_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  embeddings,
  embeddingsFinisher
))

val data = Seq("John loves apples. Mary loves oranges. John loves Mary.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.8951074481010437,0.13753940165042877,0.3108254075050354,-1.65693199634552...|
|[-0.6180210709571838,-0.12179657071828842,-0.191165953874588,-1.4497021436691...|
|[-0.822715163230896,0.7568016648292542,-0.1165061742067337,-1.59048593044281,...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[BertSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[BertSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings.scala)
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