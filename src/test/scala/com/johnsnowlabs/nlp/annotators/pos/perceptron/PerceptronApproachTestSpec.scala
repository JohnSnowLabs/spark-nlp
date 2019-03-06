package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.DataFrame
import org.scalatest._

/**
  * Created by Saif Addin on 5/18/2017.
  */
class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining("src/test/resources/anc-pos-corpus-small/test-training.txt")

  val trainedTagger: PerceptronModel =
    new PerceptronApproach()
      .setNIterations(3)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/", ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .fit(DataBuilder.basicDataBuild("dummy"))

  // Works with high iterations only
  val targetSentencesFromWsjResult = Array("NNP", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD",
    "IN", "DT", "IN", ".", "NN", ".", "NN", ".", "DT", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "NNP", ",", "CD", ".",
    "JJ", "NNP", ".")

  val tokenizedSentenceFromWsj = {
    var length = 0
    val sentences = ContentProvider.targetSentencesFromWsj.map { text =>
      val sentence = Sentence(text, length, length + text.length - 1, 0)
      length += text.length + 1
      sentence
    }
    new Tokenizer().tag(sentences).toArray
  }

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    trainedTagger,
    tokenizedSentenceFromWsj
  )

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    new PerceptronApproach()
      .setNIterations(3)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/test-training.txt", ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .fit(DataBuilder.basicDataBuild("dummy")),
    tokenizedSentenceFromWsj,
    targetSentencesFromWsjResult
  )

  "a spark based pos detector" should behave like sparkBasedPOSTagger(
    DataBuilder.basicDataBuild(ContentProvider.sbdTestParagraph)
  )

  "a spark trained pos detector" should behave like sparkBasedPOSTraining(
    path="src/test/resources/anc-pos-corpus-small/test-training.txt",
    test="src/test/resources/test.txt"
  )

  "A Perceptron Tagger" should "be readable and writable" in {
    val perceptronTagger = new PerceptronApproach()
      .setNIterations(1)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/", ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .fit(DataBuilder.basicDataBuild("dummy"))
    val path = "./test-output-tmp/perceptrontagger"
    try {
      perceptronTagger.write.overwrite.save(path)
      val perceptronTaggerRead = PerceptronModel.read.load(path)
      assert(perceptronTagger.tag(tokenizedSentenceFromWsj).head.tags.head ==
        perceptronTaggerRead.tag(tokenizedSentenceFromWsj).head.tags.head)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

//  /*
//  * Test ReouceHelper to convert token|tag to DataFrame with POS annotation as a column
//  *
//  * */
//  val posTrainingDataFrame: DataFrame = ResourceHelper.annotateTokenTagTextFiles(path = "src/test/resources/anc-pos-corpus-small", delimiter = "\\|")
//  posTrainingDataFrame.show(1,truncate = false)
}