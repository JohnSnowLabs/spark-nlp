package com.jsl.nlp.annotators.sbd.pragmatic.rule

import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticSymbols
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/8/2017.
  */
object RuleStrategy extends Enumeration {
  type Strategy = Value
  val APPEND_WITH_SYMBOL,
      PREPEND_WITH_SYMBOL,
      REPLACE_ALL_WITH_SYMBOL,
      REPLACE_WITH_SYMBOL_AND_BREAK,
      PROTECT_WITH_SYMBOL,
      REPLACE_EACH_WITH_SYMBOL,
      REPLACE_EACH_WITH_SYMBOL_AND_BREAK = Value
}

class RuleFactory(ruleStrategy: RuleStrategy.Strategy) {

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

  def applyStrategy(text: String): String = {
    ruleStrategy match {
      case PROTECT_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule.regex replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: " +
          s"'${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $PROTECT_WITH_SYMBOL")
        PragmaticSymbols.PROTECTION_MARKER_OPEN + m.matched + PragmaticSymbols.PROTECTION_MARKER_CLOSE
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def applyWith(symbol: String, text: String): String = {
    ruleStrategy match {
      case APPEND_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule.regex replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $APPEND_WITH_SYMBOL")
        "$0" + symbol
      }))
      case PREPEND_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule.regex replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $PREPEND_WITH_SYMBOL")
        symbol + "$0"
      }))
      case REPLACE_ALL_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule.regex replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $REPLACE_ALL_WITH_SYMBOL")
        symbol
      }))
      case REPLACE_WITH_SYMBOL_AND_BREAK => rules.foldRight(text)((rule, w) => rule.regex replaceAllIn(
        w, m => {
          logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule.description}' with strategy $REPLACE_WITH_SYMBOL_AND_BREAK")
          symbol + PragmaticSymbols.BREAK_INDICATOR
        }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def applySymbolicRules(text: String): String = {
    ruleStrategy match {
      case REPLACE_EACH_WITH_SYMBOL => symbolRules.foldRight(text)((rule, w) => rule._2.regex replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule._2.description}' with strategy $REPLACE_EACH_WITH_SYMBOL")
        rule._1
      }))
      case REPLACE_EACH_WITH_SYMBOL_AND_BREAK => symbolRules.foldRight(text)((rule, w) => rule._2.regex replaceAllIn(
        w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source.subSequence(
            logSubStartHelper(m.start),
            logSubEndHelper(m.source.length, m.end)
          )}' using rule: '${rule._2.description}' with strategy $REPLACE_EACH_WITH_SYMBOL_AND_BREAK")
        rule._1 + PragmaticSymbols.BREAK_INDICATOR
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

}
