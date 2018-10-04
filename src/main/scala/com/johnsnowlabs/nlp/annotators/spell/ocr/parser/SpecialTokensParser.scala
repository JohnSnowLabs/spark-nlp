package com.johnsnowlabs.nlp.annotators.spell.ocr.parser

import com.github.liblevenshtein.transducer.{Algorithm, Candidate, ITransducer}

trait TokenParser {

  def belongs(token:String):Boolean
  def splits(token:String):Seq[CandidateSplit]

  // the type of tokens this parser won't detect, but will pass the token to another parser
  val parsers:Seq[TokenParser]

  // separate the token with spaces so it can be tokenized splitting on spaces
  def separate(word:String):String
}

case class CandidateSplit(candidates:Seq[Seq[String]], cost:Int=0) {
  def appendLeft(token: String) = {
    CandidateSplit(candidates :+ Seq(token))
  }
}


object SuffixedToken extends TokenParser {

  private val suffixes = Array(",", ".", ":", "%", ";", "]", "-", "?", "'")

  private def parse(token:String)  =
    (token.dropRight(1), token.last.toString)

  override def belongs(token: String): Boolean = suffixes.map(token.endsWith).reduce(_ || _)

  override def splits(token: String): Seq[CandidateSplit] =
    if (belongs(token)) {
      val (prefix, suffix) = parse(token)
      parsers.flatMap(_.splits(prefix)).map(_.appendLeft(suffix))
    }
    else
      Seq.empty

  override val parsers: Seq[TokenParser] = Seq(DateToken, DictWord, NumberToken, RoundBrackets)

  override def separate(token:String): String ={
    var tmp = token
    suffixes.foreach{ symbol =>
      tmp = tmp.replace(symbol, s" $symbol ")
    }
    tmp
  }
}


object DateToken extends TokenParser {

  val dateRegex = ".*([0-9]{2}/[0-9]{2}/[0-9]{4}).*".r

  override def belongs(token: String): Boolean = dateRegex.pattern.matcher(token).matches

  // so far it only proposes candidates with 0 distance(the token itself)
  override def splits(token: String): Seq[CandidateSplit] ={
    if (belongs(token))
      Seq(CandidateSplit(Seq(Seq(token))))
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq.empty

  override def separate(word: String): String = {
    val matcher = dateRegex.pattern.matcher(word)
    if (matcher.matches) {
      val result = word.replace(matcher.group(1), "_DATE_")
      //println(s"$word -> $result")
      result
    }
    else
      word
  }

}

object NumberToken extends TokenParser {
  private val numRegex =
    """^?([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+\-[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+\-[0-9]+\-[0-9]+|[0-9]+)$""".r

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
      val result = word.replace(matcher.group(1), "_NUM_")
      //println(s"$word -> $result")
      result
    }
    else
      word
  }
}


class OpenCloseToken(open:String, close:String) extends TokenParser {

  override def belongs(token: String): Boolean = token.startsWith(open) && token.endsWith(close)

  override def splits(token: String): Seq[CandidateSplit] = {
    if(belongs(token))
      Seq(CandidateSplit(Seq(Seq(open), Seq(token.drop(1).dropRight(1)), Seq(close))))
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq(DateToken, NumberToken)

  override def separate(word: String): String = {
    word.
      replace(open, s" $open ").
      replace(close, s" $close ")
  }
}


class DictWord(var dict:ITransducer[Candidate]) extends TokenParser {
  import scala.collection.JavaConversions._

  def setDict(vocab:ITransducer[Candidate]) = {
    dict = vocab
  }

  override def belongs(token: String): Boolean = dict != null && dict.transduce(token).iterator().hasNext

  override def splits(token: String): Seq[CandidateSplit] = {
    if (dict != null) { // we're parsing real data
      val res = dict.transduce(token, 3).toSeq
      if (res.isEmpty)
        Seq.empty
      else {
        val min = res.minBy(_.distance()).distance
        Seq(CandidateSplit(Seq(res.filter(_.distance == min).map(_.term)), min))
      }
    } else // we're extracting the vocabulary
      Seq(CandidateSplit(Seq(Seq(token))))
  }

  override val parsers: Seq[TokenParser] = Seq.empty
  override def separate(word: String): String = word
}

// TODO fill this
object DictWord extends DictWord(null)

object RoundBrackets extends OpenCloseToken("(", ")")
object DoubleQuotes extends OpenCloseToken("\"", "\"")



object BaseParser {

  val parsers = Seq(SuffixedToken, RoundBrackets, DoubleQuotes, DateToken, NumberToken, DictWord)

  def parse(token:String):Seq[CandidateSplit] = {
      parsers.flatMap(_.splits(token))
  }
}

