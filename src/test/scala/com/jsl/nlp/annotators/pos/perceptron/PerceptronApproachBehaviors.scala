package com.jsl.nlp.annotators.pos.perceptron

import org.scalatest._

import scala.collection.mutable.{Set => MSet}

/**
  * Created by Saif Addin on 5/18/2017.
  */
trait PerceptronApproachBehaviors { this: FlatSpec =>

  def isolatedPerceptronTraining(trainingSentences: List[TaggedSentence]): Unit = {
    s"Average Perceptron tagger" should "successfully train a provided wsj corpus" in {
      val nIterations = 5
      PerceptronApproach.train(trainingSentences, nIterations)
      val tagger = PerceptronApproach.train(trainingSentences)
      val model = tagger.model
      val nWords = trainingSentences.map(_.words.length).sum
      assert(
        nWords * nIterations == model.getUpdateIterations,
        s"because Words: $nWords -- nIterations: $nIterations -- multip: ${nWords * nIterations}" +
          s"-- model iters: ${model.getUpdateIterations}"
      )
      val tagSet: MSet[String] = MSet()
      trainingSentences.foreach{s => {
        s.tags.foreach(tagSet.add)
      }}
      assert(tagSet.size == model.getTags.length)
      tagSet.foreach(tag => assert(model.getTags.contains(tag)))
    }
  }

  def isolatedPerceptronTagging(
                                 trainedTagger: PerceptronApproach,
                                 targetSentences: Array[String]
                               ): Unit = {
    s"Average Perceptron tagger" should "successfully tag all word sentences after training" in {
      val result = trainedTagger.tag(targetSentences)
      assert(result.length == targetSentences.head.split("\\W+").length, "because tagger returned less than the amount of appropriate tagged words")
      info(s"tagged words are ${result.map(t => (t.word, t.tag)).mkString("<>")}")
    }
  }

}
