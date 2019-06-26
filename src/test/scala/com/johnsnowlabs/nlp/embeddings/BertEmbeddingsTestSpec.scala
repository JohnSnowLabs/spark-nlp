package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class BertEmbeddingsTestSpec extends FlatSpec {

  "Bert Embeddings" should "correctly embed tokens and sentences" ignore {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on the notebooks, something is happening."
    ).toDF("text")

    val data1 = Seq(
      "In the Seven Kingdoms of Westeros, a soldier of the ancient Night's Watch order survives an attack by supernatural creatures known as the White Walkers, thought until now to be mythical."
    ).toDF("text")

    val data2 = Seq(
      "In King's Landing, the capital, Jon Arryn, the King's Hand, dies under mysterious circumstances."
    ).toDF("text")

    val data3 = Seq(
      "Tyrion makes saddle modifications for Bran that will allow the paraplegic boy to ride."
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BertEmbeddings.pretrained("bert_uncased", "en")
      .setInputCols("document")
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val bertDDD = pipeline.fit(ddd).transform(ddd)
    val bertDF1 = pipeline.fit(data1).transform(data1)
    val bertDF2 = pipeline.fit(data2).transform(data2)
    val bertDF3 = pipeline.fit(data3).transform(data3)

    bertDDD.show()
    bertDF3.show()
    bertDF2.show()
    bertDF1.show()

  }

}
