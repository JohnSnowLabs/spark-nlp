package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.common.{TaggedSentence, TokenizedSentence}
import com.jsl.nlp.annotators.pos.POSTagger
import com.jsl.nlp.util.io.ResourceHelper
import com.jsl.nlp.{ContentProvider, DataBuilder}
import org.scalatest._

/**
  * Created by Saif Addin on 5/18/2017.
  */
class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  val trainingSentences: Array[TaggedSentence] = ResourceHelper
    .parsePOSCorpusFromText(ContentProvider.wsjTrainingCorpus, '|')

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining(
    trainingSentences
  )

  val trainedTagger: PerceptronApproach =
    PerceptronApproach.train(nIterations = 2)

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    trainedTagger,
    ContentProvider.targetSentencesFromWsj.map(sentence => TokenizedSentence(sentence.split(" ").map(_.trim)))
  )

  val targetSentencesFromWsjResult = Array("DT","NN","IN","NN","RB","VBN","TO","VB","NNP","NN","NNS","VBZ","VBN",
    "DT","JJ","NN","IN","NN","NNS","IN","DT","NN","IN","NNS","VBN","TO","PRP","RBR","IN","CD","NNS","IN","NNS","VBD")

  val tokenizedSentenceFromWsj = ContentProvider.targetSentencesFromWsj
    .map(sentence => TokenizedSentence(sentence.split(" ").map(_.trim)))

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    PerceptronApproach.train(trainingSentences, 5),
    tokenizedSentenceFromWsj,
    targetSentencesFromWsjResult
  )

  "a spark based pragmatic detector" should behave like sparkBasedPOSTagger(
    DataBuilder.basicDataBuild(ContentProvider.sbdTestParagraph)
  )

  "A Perceptron Tagger" should "be readable and writable" in {
    val perceptronTagger = new POSTagger().setModel(PerceptronApproach.train(nIterations = 1))
    val path = "./test-output-tmp/perceptrontagger"
    try {
      perceptronTagger.write.overwrite.save(path)
      val perceptronTaggerRead = POSTagger.read.load(path)
      assert(perceptronTagger.getModel.description == perceptronTaggerRead.getModel.description)
      assert(perceptronTagger.getModel.tag(tokenizedSentenceFromWsj).head.tags.head ==
        perceptronTaggerRead.getModel.tag(tokenizedSentenceFromWsj).head.tags.head)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

}