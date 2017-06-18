package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.common.{TaggedSentence, TokenizedSentence}
import com.jsl.nlp.{ContentProvider, DataBuilder}
import com.jsl.nlp.util.ResourceHelper
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
  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    PerceptronApproach.train(trainingSentences, 5),
    ContentProvider.targetSentencesFromWsj.map(sentence => TokenizedSentence(sentence.split(" ").map(_.trim))),
    targetSentencesFromWsjResult
  )

  "a spark based pragmatic detector" should behave like sparkBasedPOSTagger(
    DataBuilder.basicDataBuild(ContentProvider.sbdTestParagraph)
  )

}