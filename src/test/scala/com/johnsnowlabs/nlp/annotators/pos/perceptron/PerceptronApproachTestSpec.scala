package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TokenizedSentence}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import org.scalatest._

/**
  * Created by Saif Addin on 5/18/2017.
  */
class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining("/anc-pos-corpus/test-training.txt")

  val trainedTagger: PerceptronModel =
    new PerceptronApproach().fit(DataBuilder.basicDataBuild("dummy"))

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    trainedTagger,
    ContentProvider.targetSentencesFromWsj.map(sentence => TokenizedSentence(sentence.split(" ").map(_.trim)))
  )


  // Works with high iterations only
  val targetSentencesFromWsjResult = Array("NNP", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD",
  "IN", "DT", "IN", ".", "NN", ".", "NN", ".", "DT", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "NNP", ",", "CD", ".",
  "JJ", "NNP", ".")

  val tokenizedSentenceFromWsj = ContentProvider.targetSentencesFromWsj
    .map(sentence => TokenizedSentence(sentence.split(" ").map(_.trim)))

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    new PerceptronApproach()
      .setNIterations(3)
      .setCorpusPath("/anc-pos-corpus/test-training.txt")
      .fit(DataBuilder.basicDataBuild("dummy")),
    tokenizedSentenceFromWsj,
    targetSentencesFromWsjResult
  )

  "a spark based pragmatic detector" should behave like sparkBasedPOSTagger(
    DataBuilder.basicDataBuild(ContentProvider.sbdTestParagraph)
  )

  "A Perceptron Tagger" should "be readable and writable" in {
    val perceptronTagger = new PerceptronApproach().setNIterations(1).fit(DataBuilder.basicDataBuild("dummy"))
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

}