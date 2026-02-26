{%- capture title -%}
M2M100Transformer
{%- endcapture -%}

{%- capture description -%}
[M2M100](https://huggingface.co/facebook/m2m100_418M) is a multilingual machine translation model developed by Facebook AI (Meta AI). Unlike previous models that require English as an intermediate language, M2M100 can directly translate between any pair of 100 languages, enabling true many-to-many translation. It is trained on a large-scale dataset covering 7.5 billion sentence pairs across 100 languages.

M2M100 supports both text translation and zero-shot translation for language pairs not seen during training. The model is available in several sizes, including 418M and 1.2B parameters.

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val m2m100 = M2M100Transformer.pretrained("m2m100_418M","xx") 
    .setInputCols(Array("documents"))
    .setMaxOutputLength(50) 
    .setOutputCol("generation") 
    .setSrcLang("en") 
    .setTgtLang("zh")
```
The default model is `"m2m100_418M"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=M2M100Transformer).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resource**:

- [M2M100 on HuggingFace](https://huggingface.co/facebook/m2m100_418M)
- [Meta AI M2M100 Announcement](https://ai.meta.com/blog/introducing-m2m-100-first-multilingual-machine-translation-model/)
- [M2M-100: Massively Multilingual Machine Translation (Paper)](https://arxiv.org/abs/2010.11125)
- [Awesome Multilingual NLP (GitHub)](https://github.com/csebuetnlp/awesome-multilingual-nlp#machine-translation)
- [Fine-Tuning M2M100: Guide (DataCamp)](https://www.datacamp.com/tutorial/m2m100-fine-tuning)

**Paper abstract**

*We introduce M2M-100, the first many-to-many multilingual translation model that can translate directly between any pair of 100 languages. Previous multilingual models rely on English-centric data and require English as an intermediate language. M2M-100 is trained on a large-scale dataset of 7.5B sentence pairs from 100 languages, enabling direct translation between any language pair. We show that M2M-100 outperforms previous models on several benchmarks and enables zero-shot translation for language pairs not seen during training.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
TRANSLATION
{%- endcapture -%}

{%- capture api_link -%}
[M2M100Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/M2M100Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[M2M100Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/m2m100_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[M2M100Transformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/M2M100Transformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import M2M100Transformer
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

m2m100 = M2M100Transformer.pretrained() \
    .setInputCols(["documents"]) \
    .setOutputCol("translation") \
    .setSourceLang("en") \
    .setTargetLang("fr") \
    .setMinOutputLength(10) \
    .setMaxOutputLength(100)

pipeline = Pipeline().setStages([
    document_assembler,
    m2m100
])

data = spark.createDataFrame([("Machine translation is a challenging task.",)], ["text"])

model = pipeline.fit(data)
results = model.transform(data)

results.select("translation.result").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val m2m100 = M2M100Transformer.pretrained()
  .setInputCols("documents")
  .setOutputCol("translation")
  .setSourceLang("en")
  .setTargetLang("fr")
  .setMinOutputLength(10)
  .setMaxOutputLength(100)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  m2m100
))

val data = Seq("Machine translation is a challenging task.").toDF("text")

val model = pipeline.fit(data)
val results = model.transform(data)

results.select("translation.result").show(false)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}