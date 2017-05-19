package com.jsl.nlp.annotators.pos.perceptron

import org.scalatest._

import scala.collection.mutable.{Set => MSet}

/**
  * Created by Saif Addin on 5/18/2017.
  */
trait PerceptronApproachBehaviors { this: FlatSpec =>

  def isolatedPerceptronTraining(tagger: PerceptronApproach, trainingSentences: List[(List[String], List[String])]): Unit = {
    s"Average Perceptron tagger" should "successfully train a provided wsj corpus" in {
      val nIterations = 5
      tagger.train(trainingSentences, nIterations)
      val nWords = trainingSentences.map { case (words, _) => words.length }.sum
      assert(
        nWords * nIterations == tagger.model.nIteration,
        s"because Words: $nWords -- nIterations: $nIterations -- multip: ${nWords * nIterations}" +
          s"-- model iters: ${tagger.model.nIteration}"
      )
      val tagSet: MSet[String] = MSet()
      trainingSentences.foreach{case (_, tags) => {
        tags.foreach(tagSet.add)
      }}
      assert(tagSet.size == tagger.model.classes.size)
      tagSet.foreach(tag => assert(tagger.model.classes.contains(tag)))
    }
  }

  def isolatedPerceptronTagging(
                                 trainedTagger: PerceptronApproach,
                                 targetSentences: Array[String]
                               ): Unit = {
    s"Average Perceptron tagger" should "successfully tag all word sentences after training" in {
      val result = trainedTagger.tag(targetSentences)
      assert(result.length == targetSentences.head.split("\\W+").length, "because tagger returned less than the amount of appropriate tagged words")
      val verbsFound = result.filter(_.tag == "VBN").map(_.word)
      val correctVerbs = Array("used", "caused", "exposed")
      assert(verbsFound.length == correctVerbs.length && verbsFound.forall(correctVerbs.contains), "because verbs are not properly tagged")
      val nounsFound = result.filter(_.tag == "NN").map(_.word)
      val correctNouns = Array("form", "asbestos", "cigarette", "percentage", "cancer", "group")
      assert(nounsFound.length == correctNouns.length && nounsFound.forall(correctNouns.contains), "because nouns are not properly tagged")
    }
  }

}
