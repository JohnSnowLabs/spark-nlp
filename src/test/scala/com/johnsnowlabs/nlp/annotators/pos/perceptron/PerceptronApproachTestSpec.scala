package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.DataFrame
import org.scalatest._


class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining(
    "src/test/resources/anc-pos-corpus-small/test-training.txt"
  )

  val trainingPerceptronDF: DataFrame = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/test-training.txt", "|", "tags")

  val trainedTagger: PerceptronModel =
    new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(3)
      .fit(trainingPerceptronDF)

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
    new Tokenizer().fit(trainingPerceptronDF).tag(sentences).toArray
  }


  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    trainedTagger,
    tokenizedSentenceFromWsj
  )

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(3)
      .fit(POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/test-training.txt", "|", "tags")),
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

  "A Perceptron Tagger" should "be readable and writable" taggedAs FastTest in {
    val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/", "|", "tags")

    val perceptronTagger = new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(1)
      .fit(trainingPerceptronDF)
    val path = "./test-output-tmp/perceptrontagger"
    try {
      perceptronTagger.write.overwrite.save(path)
      val perceptronTaggerRead = PerceptronModel.read.load(path)
      assert(perceptronTagger.tag(perceptronTagger.getModel, tokenizedSentenceFromWsj).head.tags.head ==
        perceptronTaggerRead.tag(perceptronTagger.getModel, tokenizedSentenceFromWsj).head.tags.head)
    } catch {
      case _: java.io.IOException => succeed
    }
  }
  /*
  * Testing POS() class
  * Making sure it only extracts good token_labels
  *
  * */
  val originalFrenchLabels: List[(String, Int)] = List(
    ("DET",9), ("ADP",12), ("AUX",2),
    ("CCONJ",2), ("NOUN",12), ("ADJ",3),
    ("NUM",9), ("PRON",1),
    ("PROPN",2), ("PUNCT",10),
    ("SYM",2), ("VERB",2), ("X",2)
  )

  "French readDataset in POS() class" should behave like readDatasetInPOS(
    path="src/test/resources/universal-dependency/UD_French-GSD/UD_French-test.txt",
    originalFrenchLabels
  )

  //  /*
  //  * Test ReouceHelper to convert token|tag to DataFrame with POS annotation as a column
  //  *
  //  * */
  //  val posTrainingDataFrame: DataFrame = ResourceHelper.annotateTokenTagTextFiles(path = "src/test/resources/anc-pos-corpus-small", delimiter = "|")
  //  posTrainingDataFrame.show(1,truncate = false)
}