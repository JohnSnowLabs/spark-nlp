package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import org.apache.spark.ml.Pipeline
import org.scalatest._

class ElmoEmbeddingsTestSpec extends FlatSpec {
  "Elmo Embeddings" should "generate annotations" in {
    System.out.println("Working Directory = " + System.getProperty("user.dir"))
    val data = Seq(
      "i like pancakes in the summer",
      "If I had asked people what they wanted, they would have said faster horses"
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    val embeddings = ElmoEmbeddings.loadFromPython("src/test/resources/embeddings/elmo", ResourceHelper.spark).setPoolingLayer(0)
      .setInputCols(Array("token", "document"))
      .setOutputCol("elmo")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val elmoDDD = pipeline.fit(data).transform(data)


    elmoDDD.show(false)

  }


}
