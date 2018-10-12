package com.johnsnowlabs.nlp.annotators.spell.ocr.parser

import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.github.liblevenshtein.transducer.{Algorithm, Candidate, ITransducer}
import com.johnsnowlabs.nlp.annotators.spell.ocr.TokenClasses
import com.navigamez.greex.GreexGenerator

trait TokenParser {

  val regex:String

  def generateTransducer: ITransducer[Candidate] = {
    import scala.collection.JavaConversions._

    // first step, enumerate the regular language
    val generator = new GreexGenerator(regex)
    val matches = generator.generateAll

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(matches.toList.sorted, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(2).
      includeDistance(true).
      build[Candidate]
  }

  def replaceWithLabel(tmp: String): String


  def belongs(token:String):Boolean
  def splits(token:String):Seq[CandidateSplit]

  // the type of tokens this parser won't detect, but will pass the token to another parser
  val parsers:Seq[TokenParser]

  // separate the token with spaces so it can be tokenized splitting on spaces
  def separate(word:String):String
}

case class CandidateSplit(candidates:Seq[Seq[String]], cost:Float=0f) {
  def appendLeft(token: String) = {
    CandidateSplit(candidates :+ Seq(token))
  }
}


class SuffixedToken(suffixes:Array[String]) extends TokenParser {

  private def parse(token:String)  =
    (token.dropRight(1), token.last.toString)

  override def belongs(token: String): Boolean =
    if(token.length > 1)
       suffixes.map(token.endsWith).reduce(_ || _)
    else
       false

  override def splits(token: String): Seq[CandidateSplit] =
    if (belongs(token)) {
      val (prefix, suffix) = parse(token)
      parsers.flatMap(_.splits(prefix)).map(_.appendLeft(suffix))
    }
    else
      Seq.empty

  override val parsers: Seq[TokenParser] = Seq(DateToken, NumberToken)

  override def separate(token:String): String = {
    if(belongs(token)) {
      s"""${separate(token.dropRight(1))} ${token.last}"""
    }
    else
      token
  }

  // so far we don't see a reason to replace this one
  override def replaceWithLabel(tmp: String): String = tmp
  override val regex: String = ""
}

object SuffixedToken {
  def apply(suffixes:Array[String]) = new SuffixedToken(suffixes)
}


class PrefixedToken(prefixes:Array[String]) extends TokenParser {

  private def parse(token:String)  =
    (token.head.toString, token.tail)

  override def belongs(token: String): Boolean =
    if(token.length > 1)
      prefixes.map(token.head.toString.equals).reduce(_ || _)
    else
      false

  override def splits(token: String): Seq[CandidateSplit] =
    if (belongs(token)) {
      val (prefix, suffix) = parse(token)
      parsers.flatMap(_.splits(prefix)).map(_.appendLeft(suffix))
    }
    else
      Seq.empty

  override val parsers: Seq[TokenParser] = Seq(DateToken, NumberToken)

  override def separate(token:String): String = {
    if (belongs(token))
        s"""${token.head} ${separate(token.tail)}"""
    else
        token
  }

  // so far we don't see a reason to replace this one
  override def replaceWithLabel(tmp: String): String = tmp
  override val regex: String = ""
}

object PrefixedToken {
  def apply(prefixes:Array[String]) = new PrefixedToken(prefixes)
}

object DateToken extends TokenParser with TokenClasses{

  val dateRegex = "\\(?(01|02|03|04|05|06|07|08|09|10|11|12)\\/[0-1][0-9]\\/(1|2)[0-9]{3}\\)?".r
  override val regex = "(01|02|03|04|05|06|07|08|09|10|11|12)\\/[0-1][0-9]\\/(1|2)[0-9]{3}"

  override def belongs(token: String): Boolean = dateRegex.pattern.matcher(token).matches

  // so far it only proposes candidates with 0 distance(the token itself)
  override def splits(token: String): Seq[CandidateSplit] ={
    val dist = wLevenshteinDateDist(token)
    if (dist < 3.0) {
      //val candidates = ??? -> Seq(token)
      Seq(CandidateSplit(Seq(Seq(token)), dist))
    }
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq.empty

  override def separate(word: String): String = {
    val matcher = dateRegex.pattern.matcher(word)
    if (matcher.matches) {
      val result = word.replace(matcher.group(0), "_DATE_")
      //println(s"$word -> $result")
      result
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}

object NumberToken extends TokenParser {

  private val numRegex =
    """(\-|#)?([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+\-[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+\-[0-9]+\-[0-9]+|[0-9]+)""".r

  override val regex =
    "([0-9]{1,4}\\.[0-9]{1,2}|[0-9]{1,2})"

  override def belongs(token: String): Boolean = numRegex.pattern.matcher(token).matches

  override def splits(token: String): Seq[CandidateSplit] = {
    if (belongs(token))
      Seq(CandidateSplit(Seq(Seq(token))))
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq.empty

  override def separate(word: String): String = {
    val matcher = numRegex.pattern.matcher(word)
    if(matcher.matches) {
      val result = word.replace(matcher.group(0), "_NUM_")
      //println(s"$word -> $result")
      result
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}


