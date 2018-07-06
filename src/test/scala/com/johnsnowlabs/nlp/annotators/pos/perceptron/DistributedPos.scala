package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.scalatest._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

import org.apache.spark.ml.Pipeline

class DistributedPos extends FlatSpec with PerceptronApproachBehaviors {

  "distributed pos" should "successfully work" in {

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
      .setCorpus("/home/saif/IdeaProjects/spark-nlp-models/src/main/resources/pos-corpus/anc/*", "|", "SPARK_DATASET", Map("format" -> "text"))
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
    val correct = Array("pos", " -> ", "DT", "NN", "IN", "NNS", "RB", "VBN", "TO", "VB", "NNP", "NN", "NNS", "VBZ", "VBN", "DT", "JJ", "NN", "IN", "NN", "NNS", "IN", "DT", "NN", "IN", "NNS", "VBN", "TO", "PRP", "JJR", "IN", "CD", "NNS", "RB", "NNS", "VBD")

    println(result.mapValues(_.mkString(",")).mkString(","))
    println(correct.mkString(","))

  }

}
