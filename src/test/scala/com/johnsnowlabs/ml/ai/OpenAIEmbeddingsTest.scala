package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class OpenAIEmbeddingsTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "OpenAIEmbeddings" should "generate a completion for prompts" taggedAs SlowTest in {
    // Set OPENAI_API_KEY env variable to make this work
    val promptDF = Seq("The food was delicious and the waiter...").toDS.toDF("text")

    promptDF.show(false)

    val openAIEmbeddings = new OpenAIEmbeddings()
      .setInputCols("document")
      .setOutputCol("embeddings")
      .setModel("text-embedding-ada-002")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, openAIEmbeddings))
    val completionDF = pipeline.fit(promptDF).transform(promptDF)
    completionDF.select("embeddings").show(false)
  }

}
