package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.TaggedSentence
import com.jsl.nlp.annotators.sda.SentimentApproach
import com.jsl.nlp.util.ResourceHelper
import com.typesafe.config.{Config, ConfigFactory}

/**
  * Created by saif1_000 on 16/06/2017.
  */
class PragmaticScorer(
                       sentimentDict: Map[String, String] = ResourceHelper.defaultSentDict
                     ) extends SentimentApproach {

  private case class ProcessedKey(fullKey: String, keyHead: String, keyLength: Int, sentiment: String)

  override val description = "Rule based sentiment analysis approach"
  override val requiresLemmas: Boolean = true
  override val requiresPOS: Boolean = false

  private val config: Config = ConfigFactory.load

  private val POSITIVE = "positive"
  private val NEGATIVE = "negative"
  private val INCREMENT = "increment"
  private val DECREMENT = "decrement"
  private val REVERT = "revert"

  private val POSITIVE_VALUE = config.getDouble("nlp.sentimentParams.positive")
  private val NEGATIVE_VALUE = config.getDouble("nlp.sentimentParams.negative")
  private val INCREMENT_MULTIPLIER = config.getDouble("nlp.sentimentParams.increment")
  private val DECREMENT_MULTIPLIER = config.getDouble("nlp.sentimentParams.decrement")
  private val REVERT_MULTIPLIER = config.getDouble("nlp.sentimentParams.revert")

  private val processedKeys: Array[ProcessedKey] = sentimentDict.map{case (key, sentiment) =>
    val keySplits = key.split(" ").map(_.trim)
    ProcessedKey(key, keySplits.head, keySplits.length, sentiment)
  }.toArray

  override def score(taggedSentences: Array[TaggedSentence]): Double = {
    val sentenceSentiments: Array[Array[String]] = taggedSentences.map(taggedSentence => {
      taggedSentence.taggedWords.flatMap(taggedWord => {
        processedKeys.find(processedKey => {
          taggedWord.word.toLowerCase == processedKey.keyHead &&
          taggedSentence.words.take(processedKey.keyLength).mkString(" ") == processedKey.fullKey
        }).map(processedKey => {
          sentimentDict(processedKey.fullKey)
        })
      })
    })
    val sentimentBaseScore: Array[(Array[String], Double)] = sentenceSentiments.map(sentiment => (
      sentiment,
      sentiment.foldRight(0.0)((sentiment, score) => {
        sentiment match {
          case POSITIVE => score + POSITIVE_VALUE
          case NEGATIVE => score - NEGATIVE_VALUE
          case _ => score
        }})
      ))
    sentimentBaseScore.map{case (sentiments, baseScore) => sentiments.foldRight(baseScore)((sentiment, currentScore) => {
      sentiment match {
        case INCREMENT => currentScore * INCREMENT_MULTIPLIER
        case DECREMENT => currentScore * DECREMENT_MULTIPLIER
        case REVERT => currentScore * REVERT_MULTIPLIER
        case _ => currentScore
      }
    })}.sum
  }

}
