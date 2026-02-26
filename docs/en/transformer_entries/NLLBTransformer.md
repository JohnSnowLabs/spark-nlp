{%- capture title -%} NLLBTransformer {%- endcapture -%} 
{%- capture description -%} 
NLLBTransformer is a Spark NLP annotator that leverages the No Language Left Behind (NLLB) models by Meta AI. These models are designed to provide high-quality machine translation for over 200 languages, especially low-resource ones. They are trained on massive multilingual datasets and optimized for efficient inference, making them suitable for real-world translation systems at scale.

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val seq2seq = NLLBTransformer.pretrained("nllb_distilled_600M_8int","xx") 
    .setInputCols(Array("documents")) 
    .setOutputCol("generation")
```
The default model is `"nllb_distilled_600M_8int"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=NLLBTransformer).

Spark NLP also supports Hugging Face transformer-based translation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [NLLB Project on GitHub](https://github.com/facebookresearch/fairseq/tree/nllb)  
- [NLLB: No Language Left Behind Paper](https://arxiv.org/abs/2207.04672)  
- [Meta AI NLLB Blog Post](https://ai.meta.com/research/no-language-left-behind/)  

**Paper abstract**

*Driven by the goal of eradicating language barriers on a global scale, machine translation has solidified itself as a key focus of artificial intelligence research today. However, such efforts have coalesced around a small subset of languages, leaving behind the vast majority of mostly low-resource languages. What does it take to break the 200 language barrier while ensuring safe, high quality results, all while keeping ethical considerations in mind? In No Language Left Behind, we took on this challenge by first contextualizing the need for low-resource language translation support through exploratory interviews with native speakers. Then, we created datasets and models aimed at narrowing the performance gap between low and high-resource languages. More specifically, we developed a conditional compute model based on Sparsely Gated Mixture of Experts that is trained on data obtained with novel and effective data mining techniques tailored for low-resource languages. We propose multiple architectural and training improvements to counteract overfitting while training on thousands of tasks. Critically, we evaluated the performance of over 40,000 different translation directions using a human-translated benchmark, Flores-200, and combined human evaluation with a novel toxicity benchmark covering all languages in Flores-200 to assess translation safety. Our model achieves an improvement of 44% BLEU relative to the previous state-of-the-art, laying important groundwork towards realizing a universal translation system. Finally, we open source all contributions described in this work, accessible at [this https URL.](https://github.com/facebookresearch/fairseq/tree/nllb)*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENTS
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[NLLBTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/NLLBTransformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[NLLBTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/nllb_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[NLLBTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/NLLBTransformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import NLLBTransformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("documents")   

nllb = NLLBTransformer.pretrained("nllb_distilled_600M_8int", "xx") \
    .setInputCols(["documents"]) \
    .setOutputCol("generation") \
    .setSrcLang("eng_Latn") \
    .setTgtLang("zho_Hans") \
    .setMaxOutputLength(50)

pipeline = Pipeline().setStages([
    documentAssembler, 
    nllb
])

data = spark.createDataFrame([
    ["Artificial intelligence is transforming the way people communicate, learn, and work across the world."]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show()

+-----------------------------------------------+
|result                                         |
+-----------------------------------------------+
|[人工智能正在改变人们在世界各地的沟通,学习和工作方式.] |
+-----------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.NLLBTransformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val nllb = NLLBTransformer.pretrained("nllb_distilled_600M_8int", "xx")
  .setInputCols("documents")
  .setOutputCol("generation")
  .setSrcLang("eng_Latn")
  .setTgtLang("zho_Hans")
  .setMaxOutputLength(50)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  nllb
))

val data = Seq("Artificial intelligence is transforming the way people communicate, learn, and work across the world.").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show(truncate = false)

+-----------------------------------------------+
|result                                         |
+-----------------------------------------------+
|[人工智能正在改变人们在世界各地的沟通,学习和工作方式.] |
+-----------------------------------------------+
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