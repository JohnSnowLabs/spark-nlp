package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.{WritableAnnotatorComponent, TaggedSentence}
import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent
import com.jsl.nlp.annotators.sda.SentimentApproach
import com.typesafe.config.{Config, ConfigFactory}
import org.slf4j.LoggerFactory

/**
  * Created by saif on 16/06/2017.
  */

/**
  * Scorer is a rule based implementation inspired on http://fjavieralba.com/basic-sentiment-analysis-with-python.html
  * Its strategy is to tag words by a dictionary in a sentence context, and later identify such context to get amplifiers
  * @param sentimentDict This scorer requires a dictionary of good or bad words
  */
class PragmaticScorer(sentimentDict: Map[String, String]) extends SentimentApproach {

  private val logger = LoggerFactory.getLogger("PragmaticScorer")

  /**
    * Internal class to summarize dictionary key information
    * @param fullKey represents the whole dictionary key which can be a phrase
    * @param keyHead represents the first word of the phrase, to be issued for matching
    * @param keyLength represents the amount of words in a phrase key
    * @param sentiment holds the tag for this phrase
    */
  private case class ProcessedKey(fullKey: String, keyHead: String, keyLength: Int, sentiment: String)

  override val description = "Rule based sentiment analysis approach"
  override val requiresLemmas: Boolean = true
  override val requiresPOS: Boolean = false

  /** Hardcoded tags for this implementation */
  private val POSITIVE = "positive"
  private val NEGATIVE = "negative"
  private val INCREMENT = "increment"
  private val DECREMENT = "decrement"
  private val REVERT = "revert"

  /** config is used for tunning values for tags */
  private val config: Config = ConfigFactory.load

  private val POSITIVE_VALUE = config.getDouble("nlp.sentimentParams.positive")
  private val NEGATIVE_VALUE = config.getDouble("nlp.sentimentParams.negative")
  private val INCREMENT_MULTIPLIER = config.getDouble("nlp.sentimentParams.increment")
  private val DECREMENT_MULTIPLIER = config.getDouble("nlp.sentimentParams.decrement")
  private val REVERT_MULTIPLIER = config.getDouble("nlp.sentimentParams.revert")

  /** reads the dictionary and processes it into useful information on a [[ProcessedKey]] */
  private val processedKeys: Array[ProcessedKey] = {
    sentimentDict.map{case (key, sentiment) =>
      val keySplits = key.split(" ").map(_.trim)
      ProcessedKey(key, keySplits.head, keySplits.length, sentiment)
    }.toArray
  }

  /** converts into a writable representation of [[com.jsl.nlp.annotators.sda.pragmatic.PragmaticScorer]] */
  override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = {
    SerializedScorerApproach(sentimentDict)
  }

  /**
    * scores lowercased words by their headers within a sentence context. Sets the tag for every word
    * @param taggedSentences POS tagged sentence. POS tags are not really useful in this implementation for now.
    * @return
    */
  override def score(taggedSentences: Array[TaggedSentence]): Double = {
    val sentenceSentiments: Array[Array[String]] = taggedSentences.map(taggedSentence => {
      taggedSentence.taggedWords.flatMap(taggedWord => {
        val targetWord = taggedWord.word.toLowerCase
        val targetWordPosition = taggedSentence.words.indexOf(taggedWord.word)
        processedKeys.find(processedKey => {
          /**takes the phrase based on the dictionary key phrase length to check if it matches the entire phrase*/
          targetWord == processedKey.keyHead &&
          taggedSentence.words
            .slice(targetWordPosition, targetWordPosition + processedKey.keyLength)
            .mkString(" ") == processedKey.fullKey
        }).map(processedKey => {
          /** if it exists, put the appropiate tag of the full phrase */
          logger.debug(s"matched sentiment ${processedKey.fullKey} as ${processedKey.sentiment}")
          sentimentDict(processedKey.fullKey)
        })
      })
    })
    /** score is returned based on config tweaked parameters */
    val sentimentBaseScore: Array[(Array[String], Double)] = sentenceSentiments.map(sentiment => (
      sentiment,
      sentiment.foldRight(0.0)((sentiment, score) => {
        sentiment match {
          case POSITIVE => score + POSITIVE_VALUE
          case NEGATIVE => score + NEGATIVE_VALUE
          case _ => score
        }})
      ))
    logger.debug(s"sentiment positive/negative base score is ${sentimentBaseScore.map(_._2).sum}")
    /** amplifiers alter the base score of positive and negative tags. Sums the entire sentence score */
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
