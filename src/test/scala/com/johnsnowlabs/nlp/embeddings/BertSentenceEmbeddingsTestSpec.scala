package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class BertSentenceEmbeddingsTestSpec extends FlatSpec {

  "BertSentenceEmbeddings" should "correctly work with empty tokens" ignore {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my forth.")
    )).toDF("id", "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128")
      .setInputCols("document")
      .setOutputCol("bert")
      .setCaseSensitive(false)
      .setMaxSentenceLength(64)


    val pipeline = new Pipeline().setStages(Array(document, sentence, embeddings))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select("bert.embeddings").show()

  }

}
