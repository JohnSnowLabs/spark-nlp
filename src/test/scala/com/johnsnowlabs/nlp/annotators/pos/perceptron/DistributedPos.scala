package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.ContentProvider
import org.scalatest._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.util.Benchmark
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

    val pos = new PerceptronApproachDistributed()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setCorpus("src/test/resources/anc-pos-corpus-small/*", "|", "SPARK", Map("format" -> "text"))
      .setNIterations(5)

      """
    val posLegacy = new PerceptronApproach()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setCorpus("./pos-corpus/anc/*", "|", "SPARK", Map("format" -> "text", "repartition" -> "2"))
      .setNIterations(3)
      """

    val posLegacy = PerceptronModel.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val finisher = new Finisher()
      .setInputCols("pos")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        pos,
        finisher
      ))

    val pipelineLegacy = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        posLegacy,
        finisher
      ))

    val m = Benchmark.time("Training time for distributed pos") {pipeline.fit(Seq.empty[String].toDF("text"))}
    val ml = Benchmark.time("Training time for legacy pos") {pipelineLegacy.fit(Seq.empty[String].toDF("text"))}
    val lp = new LightPipeline(m)
    val lpl = new LightPipeline(ml)

    val result = lp.annotate("A form of asbestos once used to make Kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than 30 years ago researchers reported")
    val resultLegacy = lpl.annotate("A form of asbestos once used to make Kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than 30 years ago researchers reported")
    val correct = Array("pos -> DT", "NN", "IN", "NNS", "RB", "VBN", "TO", "VB", "NNP", "NN", "NNS", "VBZ", "VBN", "DT", "JJ", "NN", "IN", "NN", "NNS", "IN", "DT", "NN", "IN", "NNS", "VBN", "TO", "PRP", "JJR", "IN", "CD", "NNS", "RB", "NNS", "VBD")

    println(result.mapValues(_.mkString(",")).mkString(","))
    println(resultLegacy.mapValues(_.mkString(",")).mkString(","))
    println(correct.mkString(","))

    println("VS Legacy accuracy ratio: ")
    println(result.values.head.zip(resultLegacy.values.head).count{case (a, b) => a == b} / result.values.head.length.toDouble)

    println("VS Legacy accuracy ratio in paragraph: ")
    val result2 = lp.annotate(ContentProvider.sbdTestParagraph.replaceAll("@@", ""))
    val resultLegacy2 = lpl.annotate(ContentProvider.sbdTestParagraph.replaceAll("@@", ""))
    println(result2.values.head.zip(resultLegacy2.values.head).count{case (a, b) => a == b} / result2.values.head.length.toDouble)

    val result3 = lp.annotate("Available loan amounts ranges from US $100,000.00 to US $5,000,000.00 with repayment duration of 1 to 10 years.")
    val resultLegacy3 = lpl.annotate("Available loan amounts ranges from US $100,000.00 to US $5,000,000.00 with repayment duration of 1 to 10 years.")
    val correct3 = Array("pos -> NNP", "NN", "VBZ", "NNS", "IN", "NNP", "$", "CD", "TO", "NNP", "$", "CD", "IN", "NN", "NN", "IN", "CD", "TO", "CD", "NNS", ".")

    println(result3.mapValues(_.mkString(",")).mkString(","))
    println(resultLegacy3.mapValues(_.mkString(",")).mkString(","))
    println(correct3.mkString(","))

    val result4 = lp.annotate("Ever since I started on your herbal supplement, Sharon says sex is so much more pleasurable for her, and she comes much more easily.")
    val resultLegacy4 = lpl.annotate("Ever since I started on your herbal supplement, Sharon says sex is so much more pleasurable for her, and she comes much more easily.")
    val correct4 = Array("pos -> RB", "IN", "PRP", "VBD", "IN", "PRP$", "JJ", "NN", ",", "NNP", "VBZ", "NN", "VBZ", "RB", "RB", "RBR", "JJ", "IN", "PRP$", ",", "CC", "PRP", "VBZ", "RB", "RBR", "RB", ".")

    println(result4.mapValues(_.mkString(",")).mkString(","))
    println(resultLegacy4.mapValues(_.mkString(",")).mkString(","))
    println(correct4.mkString(","))
  }

}
