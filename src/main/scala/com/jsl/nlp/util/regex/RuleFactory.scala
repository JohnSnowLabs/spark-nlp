package com.jsl.nlp.util.regex

import com.jsl.nlp.annotators.sbd.pragmatic.RuleSymbols
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import scala.util.matching.Regex

case class RegexMatch(content: String, start: Int, end: Int, description: String)

/**
  * Created by Saif Addin on 5/8/2017.
  */
object RuleStrategy extends Enumeration {

  type TransformStrategy = Value
  val NO_TRANSFORM,
      APPEND_WITH_SYMBOL,
      PREPEND_WITH_SYMBOL,
      REPLACE_ALL_WITH_SYMBOL,
      REPLACE_WITH_SYMBOL_AND_BREAK,
      PROTECT_FROM_BREAK,
      REPLACE_EACH_WITH_SYMBOL,
      REPLACE_EACH_WITH_SYMBOL_AND_BREAK = Value

  type MatchStrategy = Value
    val MATCH_ALL,
        MATCH_FIRST,
        MATCH_COMPLETE = Value

}

class RuleFactory(matchStrategy: RuleStrategy.MatchStrategy)
                 (transformStrategy: RuleStrategy.TransformStrategy) extends RuleSymbols {

  import RuleStrategy._

  val logger = Logger(LoggerFactory.getLogger("RuleFactory"))

  private var rules: Seq[RegexRule] = Seq()
  private var symbolRules: Seq[(String, RegexRule)] = Seq()

  private def logSubStartHelper(start: Int): Int = if (start > 10) start - 10 else  0
  private def logSubEndHelper(sourceLength: Int, end: Int): Int = if (sourceLength - end > 10) end + 10 else sourceLength

  def addRule(rule: RegexRule): this.type = {
    rules = rules :+ rule
    this
  }

  def addSymbolicRule(symbol: String, rule: RegexRule): this.type = {
    symbolRules = symbolRules :+ (symbol, rule)
    this
  }

  def addRules(newRules: Seq[RegexRule]): this.type = {
    rules = rules ++: newRules
    this
  }

  def setRules(newRules: Seq[RegexRule]): this.type = {
    rules = newRules
    this
  }

  def find(text: String): Seq[RegexMatch] = {
    matchStrategy match {
      case MATCH_ALL => rules.flatMap(rule => rule.regex.findAllMatchIn(text).map(m =>
        RegexMatch(m.matched, m.start, m.end, rule.description)))
      case MATCH_FIRST => rules.flatMap(rule => rule.regex.findFirstMatchIn(text).map(m =>
        RegexMatch(m.matched, m.start, m.end, rule.description)))
      case MATCH_COMPLETE => rules.flatMap(rule => rule.regex.findFirstMatchIn(text).filter(_.matched == text).map(m =>
        RegexMatch(m.matched, m.start, m.end, rule.description)))
    }
  }

  private def transformMatch(text: String, regex: Regex)(transform: Regex.Match => String): String = {
    matchStrategy match {
      case MATCH_ALL => regex.replaceAllIn(text, transform)
      case MATCH_FIRST => regex.replaceFirstIn(text, transform())
      case MATCH_COMPLETE => regex.findFirstMatchIn(text).filter(_.matched == text).map(_ =>
        regex.replaceFirstIn(text, String)).getOrElse(text)
      case _ => throw new IllegalArgumentException("Invalid match strategy")
    }
  }

  def transform(text: String): String = {
    transformStrategy match {
      case PROTECT_FROM_BREAK => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
        logger.debug(s"Matched: '${m.matched}' from: " +
          s"'${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $PROTECT_FROM_BREAK")
        PROTECTION_MARKER_OPEN + m.matched + PROTECTION_MARKER_CLOSE
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def transformWithSymbol(symbol: String, text: String): String = {
    transformStrategy match {
      case APPEND_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $APPEND_WITH_SYMBOL")
        "$0" + symbol
      }))
      case PREPEND_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $PREPEND_WITH_SYMBOL")
        symbol + "$0"
      }))
      case REPLACE_ALL_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $REPLACE_ALL_WITH_SYMBOL")
        symbol
      }))
      case REPLACE_WITH_SYMBOL_AND_BREAK => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
          logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $REPLACE_WITH_SYMBOL_AND_BREAK")
          symbol + BREAK_INDICATOR
        }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def transformWithSymbolicRules(text: String): String = {
    transformStrategy match {
      case REPLACE_EACH_WITH_SYMBOL => symbolRules.foldRight(text)((rule, target) => transformMatch(target, rule._2.regex)({ m =>
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule._2.description}' with strategy $REPLACE_EACH_WITH_SYMBOL")
        rule._1
      }))
      case REPLACE_EACH_WITH_SYMBOL_AND_BREAK => symbolRules.foldRight(text)((rule, target) => rule._2.regex replaceAllIn(
        target, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule._2.description}' with strategy $REPLACE_EACH_WITH_SYMBOL_AND_BREAK")
        rule._1 + BREAK_INDICATOR
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

}
