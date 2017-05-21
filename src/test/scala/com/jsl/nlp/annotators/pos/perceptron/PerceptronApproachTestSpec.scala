package com.jsl.nlp.annotators.pos.perceptron

import org.scalatest._

import scala.collection.mutable.ListBuffer

/**
  * Created by Saif Addin on 5/18/2017.
  */
class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  private def readTagged(text: String, sep: Char = '|'): List[(List[String], List[String])] = {
    val sentences: ListBuffer[(List[String], List[String])] = ListBuffer()
    text.split("\n").foreach{sentence => {
      val tokens: ListBuffer[String] = ListBuffer()
      val tags: ListBuffer[String] = ListBuffer()
      sentence.split("\\s+").foreach{token => {
        val tagSplit: Array[String] = token.split(sep)
        val word = tagSplit(0)
        val pos = tagSplit(1)
        tokens.append(word)
        tags.append(pos)
      }}
      sentences.append((tokens.toList, tags.toList))
    }}
    sentences.toList
  }

  val wsjTrainingCorpus: String = "Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ ,|, will|MD " +
    "join|VB the|DT board|NN as|IN a|DT nonexecutive|JJ director|NN " +
    "Nov.|NNP 29|CD .|.\nMr.|NNP Vinken|NNP is|VBZ chairman|NN of|IN " +
    "Elsevier|NNP N.V.|NNP ,|, the|DT Dutch|NNP publishing|VBG " +
    "group|NN .|. Rudolph|NNP Agnew|NNP ,|, 55|CD years|NNS old|JJ " +
    "and|CC former|JJ chairman|NN of|IN Consolidated|NNP Gold|NNP " +
    "Fields|NNP PLC|NNP ,|, was|VBD named|VBN a|DT nonexecutive|JJ " +
    "director|NN of|IN this|DT British|JJ industrial|JJ conglomerate|NN " +
    ".|.\nA|DT form|NN of|IN asbestos|NN once|RB used|VBN to|TO make|VB " +
    "Kent|NNP cigarette|NN filters|NNS has|VBZ caused|VBN a|DT high|JJ " +
    "percentage|NN of|IN cancer|NN deaths|NNS among|IN a|DT group|NN " +
    "of|IN workers|NNS exposed|VBN to|TO it|PRP more|RBR than|IN " +
    "30|CD years|NNS ago|IN ,|, researchers|NNS reported|VBD .|."

  val trainingSentences: List[(List[String], List[String])] = readTagged(wsjTrainingCorpus)

  val text = "Simple is better than complex. Complex is better than complicated"
  val tagger = new PerceptronApproach

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining(
    tagger,
    trainingSentences.map(s => TaggedSentence(s._1.map(Word), s._2))
  )

  val targetSentencesFromWsj = Array("A form of asbestos once used to make " +
    "Kent cigarette filters has caused a high percentage of cancer deaths among a group " +
    "of workers exposed to it more than 30 years ago, researchers reported")
  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    tagger,
    targetSentencesFromWsj
  )

}
