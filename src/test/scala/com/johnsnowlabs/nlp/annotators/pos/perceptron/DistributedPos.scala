package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.scalatest._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

import org.apache.spark.ml.Pipeline

class DistributedPos extends FlatSpec with PerceptronApproachBehaviors {

  "distributed pos" should "successfully work" ignore {

    import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val pos = new PerceptronApproach()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setCorpus("./pos-corpus/anc/*", "|", "SPARK_DATASET", Map("format" -> "text"))
      .setNIterations(5)

    val finisher = new Finisher()
      .setInputCols("pos")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        pos,
        finisher
      ))

    val m = pipeline.fit(Seq.empty[String].toDF("text"))
    val lp = new LightPipeline(m)

    val result = lp.annotate("A form of asbestos once used to make Kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than 30 years ago researchers reported")

    println(result.mapValues(_.mkString(",")).mkString(","))

  }

}
