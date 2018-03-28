package com.johnsnowlabs.nlp.util.regex

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.RuleSymbols
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/8/2017.
  */

/**
  * Regular Expressions rule manager. Applies rules based on Matching and Replacement strategies
  * @param matchStrategy How to decide on regex search
  * @param transformStrategy How to decide when replacing or transforming content with Regex
  */
class RuleFactory(matchStrategy: MatchStrategy.MatchStrategy,
                  transformStrategy: TransformStrategy.TransformStrategy = TransformStrategy.NO_TRANSFORM)
  extends RuleSymbols with Serializable {

  /**
    * Internal representation of a regex match
    * @param content the matching component, which holds [[Regex.Match]] information, plus its user identification
    * @param identifier user provided identification of a rule
    */
  protected case class RuleMatch(content: Regex.Match, identifier: String)

  import TransformStrategy._
  import MatchStrategy._

  /** Helper functions to identify context in a word for debugging */
  private val logger = LoggerFactory.getLogger("RuleFactory")
  private def logSubStartHelper(start: Int): Int = if (start > 10) start - 10 else  0
  private def logSubEndHelper(sourceLength: Int, end: Int): Int = if (sourceLength - end > 10) end + 10 else sourceLength

  /** Rules and SymbolRules are key pieces of regex transformation */
  private var rules: Seq[RegexRule] = Seq()
  private var symbolRules: Seq[(String, RegexRule)] = Seq()

  /** Adds a rule to this factory*/
  def addRule(rule: RegexRule): this.type = {
    rules = rules :+ rule
    this
  }

  /** Adds a rule to this factory with native types */
  def addRule(rule: Regex, description: String): this.type = {
    rules = rules :+ new RegexRule(rule, description)
    this
  }

  def clearRules(): this.type = {
    rules = Seq.empty[RegexRule]
    this
  }

  /** Shortcut functions, no need to execute them on runtime since a strategy won't change in lifetime of Factory */
  private val findMatchFunc = (text: String) => matchStrategy match {
    case MATCH_ALL => rules.flatMap(rule => rule.regex.findAllMatchIn(text).map(m => RuleMatch(m, rule.identifier)))
    case MATCH_FIRST => rules.flatMap(rule => rule.regex.findFirstMatchIn(text).map(m => RuleMatch(m, rule.identifier)))
    case MATCH_COMPLETE => rules.flatMap(rule => rule.regex.findFirstMatchIn(text).filter(_.matched == text).map(m => RuleMatch(m, rule.identifier)))
  }

  private val transformMatchFunc = (text: String, regex: Regex, transform: Regex.Match => String) => matchStrategy match {
    case MATCH_ALL => regex.replaceAllIn(text, transform)
    case MATCH_FIRST => regex.findFirstMatchIn(text).map(m => regex.replaceFirstIn(text, transform(m))).getOrElse(text)
    case MATCH_COMPLETE => regex.findFirstMatchIn(text).filter(_.matched == text).map(m =>
      regex.replaceFirstIn(text, transform(m))).getOrElse(text)
    case _ => throw new IllegalArgumentException("Invalid match strategy")
  }

  private val transformWithSymbolFunc = (symbol: String, text: String) => transformStrategy match {
    case APPEND_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start),logSubEndHelper(m.source.length, m.end)),
        () => rule.identifier,
        () => APPEND_WITH_SYMBOL)
      "$0" + symbol
    }))
    case PREPEND_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start),logSubEndHelper(m.source.length, m.end)),
        () => rule.identifier,
        () => PREPEND_WITH_SYMBOL)
      symbol + "$0"
    }))
    case REPLACE_ALL_WITH_SYMBOL => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start), logSubEndHelper(m.source.length, m.end)),
        () => rule.identifier,
        () => REPLACE_ALL_WITH_SYMBOL)
      symbol
    }))
    case REPLACE_WITH_SYMBOL_AND_BREAK => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start), logSubEndHelper(m.source.length, m.end)),
        () => rule.identifier,
        () => REPLACE_WITH_SYMBOL_AND_BREAK)
      symbol + BREAK_INDICATOR
    }))
    case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
  }

  private val transformWithSymbolicRulesFunc = (text: String) => transformStrategy match {
    case REPLACE_EACH_WITH_SYMBOL => symbolRules.foldRight(text)((rule, target) => transformMatch(target, rule._2.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start), logSubEndHelper(m.source.length, m.end)),
        () => rule._2.identifier,
        () => REPLACE_EACH_WITH_SYMBOL)
      rule._1
    }))
    case REPLACE_EACH_WITH_SYMBOL_AND_BREAK => symbolRules.foldRight(text)((rule, target) => rule._2.regex replaceAllIn(
      target, m => {
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start), logSubEndHelper(m.source.length, m.end)),
        () => rule._2.identifier,
        () => REPLACE_EACH_WITH_SYMBOL_AND_BREAK)
      rule._1 + BREAK_INDICATOR
    }))
    case PROTECT_FROM_BREAK => rules.foldRight(text)((rule, target) => transformMatch(target, rule.regex)({ m =>
      logger.debug("Matched: {} from: {} using rule {} with strategy {}",
        () => m.matched,
        () => m.source.subSequence(logSubStartHelper(m.start), logSubEndHelper(m.source.length, m.end)),
        () => rule.identifier,
        () => PROTECT_FROM_BREAK)
      PROTECTION_MARKER_OPEN + m.matched + PROTECTION_MARKER_CLOSE
    }))
    case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
  }

  /**
    * Adds a rule and its associated symbol to apply some transformation using such symbol
    * @param symbol symbol is a character to be used in a transformation application, where many rules can apply different transformations
    * @param rule rule to be used when replacing a match with a symbol
    * @return
    */
  def addSymbolicRule(symbol: String, rule: RegexRule): this.type = {
    symbolRules = symbolRules :+ (symbol, rule)
    this
  }

  /** add multiple rules alltogether */
  def addRules(newRules: Seq[RegexRule]): this.type = {
    rules = rules ++: newRules
    this
  }

  /** overrides rules with a new set of rules */
  def setRules(newRules: Seq[RegexRule]): this.type = {
    rules = newRules
    this
  }

  /**Applies factory match strategy to find matches and returns any number of Matches*/
  def findMatch(text: String): Seq[RuleMatch] = {
    findMatchFunc(text)
  }

  /** Specifically finds a first match within a group of matches */
  def findMatchFirstOnly(text: String): Option[RuleMatch] = {
    findMatch(text).headOption
  }

  /**
    * Applies rule transform strategy and utilizing matching strategies
    * Arguments are curried so transformation can be partially applied in some cases
    * @return Resulting transformation
    */
  private def transformMatch(text: String, regex: Regex)(transform: Regex.Match => String): String = {
    transformMatchFunc(text: String, regex: Regex, transform: Regex.Match => String)
  }

  /**
    * Applies factory transform of all ordered rules utilizing transform and match strategies with provided symbol
    * @param symbol a symbol to use for all transformations altogether
    * @param text target text to transform
    * @return
    */
  def transformWithSymbol(symbol: String, text: String): String = {
    transformWithSymbolFunc(symbol, text)
  }

  /**
    * Applies factory transform of all ordered rules utilizing transform and match strategies corresponding each rule with its symbol
    * @param text target text to transform
    * @return Returns a transformed text
    */
  def transformWithSymbolicRules(text: String): String = {
    transformWithSymbolicRulesFunc(text)
  }
}
object RuleFactory {
  /**Specific partial constructor for [[RuleFactory]] where MatchStrategy might change on runtime */
  def lateMatching(transformStrategy: TransformStrategy.TransformStrategy)
                  (matchStrategy: MatchStrategy.MatchStrategy): RuleFactory =
    new RuleFactory(matchStrategy, transformStrategy)
}

/**
  * Allowed strategies for [[RuleFactory]] applications regarding replacement
  */
object TransformStrategy extends Enumeration {
  type TransformStrategy = Value
  val NO_TRANSFORM,
  APPEND_WITH_SYMBOL,
  PREPEND_WITH_SYMBOL,
  REPLACE_ALL_WITH_SYMBOL,
  REPLACE_WITH_SYMBOL_AND_BREAK,
  PROTECT_FROM_BREAK,
  REPLACE_EACH_WITH_SYMBOL,
  REPLACE_EACH_WITH_SYMBOL_AND_BREAK = Value
}

/**
  * Allowed strategies for [[RuleFactory]] applications regarding matching
  */
object MatchStrategy extends Enumeration {
  type MatchStrategy = Value
  val MATCH_ALL,
  MATCH_FIRST,
  MATCH_COMPLETE = Value
}