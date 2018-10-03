package com.johnsnowlabs.nlp.annotators.spell.ocr.parser

import com.github.liblevenshtein.transducer.{Algorithm, ITransducer}
import com.github.liblevenshtein.transducer.factory.TransducerBuilder


trait TokenParser {

  def belongs(token:String):Boolean
  def splits(token:String):Seq[CandidateSplit]

  // the type of tokens this parser won't detect, but will pass the token to another parser
  val parsers:Seq[TokenParser]
}

case class CandidateSplit(candidates:Seq[Seq[String]]) {
  def appendLeft(token: String) = {
    CandidateSplit(candidates :+ Seq(token))
  }
}


object SuffixedToken extends TokenParser {

  private val suffixes = Array(",", ".", ":", "%")

  private def parse(token:String)  =
    (token.slice(0, token.length -1), token.last.toString)

  override def belongs(token: String): Boolean = suffixes.map(token.endsWith).reduce(_ || _)

  override def splits(token: String): Seq[CandidateSplit] =
    if (belongs(token)) {
      val (prefix, suffix) = parse(token)
      parsers.flatMap(_.splits(prefix)).map(_.appendLeft(suffix))
    }
    else
      Seq.empty

  override val parsers: Seq[TokenParser] = Seq(DateToken, DictWord, NumberToken, RoundBrackets)
}


object DateToken extends TokenParser {

  val dateRegex = "^([0-9]{2}/[0-9]{2}/[0-9]{4})$".r

  override def belongs(token: String): Boolean = dateRegex.pattern.matcher(token).matches

  // so far it only proposes candidates with 0 distance(the token itself)
  override def splits(token: String): Seq[CandidateSplit] ={
    if (belongs(token))
      Seq(CandidateSplit(Seq(Seq(token))))
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq.empty
}

object NumberToken extends TokenParser {
  private val numRegex =
    """^([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+\-[0-9]+|[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+\-[0-9]+\-[0-9]+)$""".r

  override def belongs(token: String): Boolean = numRegex.pattern.matcher(token).matches

  override def splits(token: String): Seq[CandidateSplit] = {
    if (belongs(token))
      Seq(CandidateSplit(Seq(Seq(token))))
    else
      Seq.empty
  }

  override val parsers: Seq[TokenParser] = Seq.empty
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
}

object DatesToken extends TokenParser {

  override def belongs(token: String): Boolean = true
  override def splits(token: String): Seq[CandidateSplit] = Seq.empty
  override val parsers: Seq[TokenParser] = Seq(DateToken, NumberToken)
}


class DictWord(var dict:ITransducer[String]) extends TokenParser {
  import scala.collection.JavaConversions._

  def setDict(vocab:Seq[String]) = {
    dict = new TransducerBuilder().
      dictionary(vocab.sorted, true).
      algorithm(Algorithm.TRANSPOSITION).
      defaultMaxDistance(2).
      includeDistance(true).
      build[String]
  }

  override def belongs(token: String): Boolean = dict != null && dict.transduce(token).iterator().hasNext

  override def splits(token: String): Seq[CandidateSplit] = {
    if (dict != null) { // we're parsing real data
      val res = dict.transduce(token).toSeq
      if (res.isEmpty)
        Seq.empty
      else
        Seq(CandidateSplit(Seq(res)))
    } else // we're extracting the vocabulary
      Seq(CandidateSplit(Seq(Seq(token))))
  }

  override val parsers: Seq[TokenParser] = Seq.empty

}

// TODO fill this
object DictWord extends DictWord(null)

object RoundBrackets extends OpenCloseToken("(", ")")
object DoubleQuotes extends OpenCloseToken("\"", "\"")



object BaseParser {

  val parsers = Seq(SuffixedToken, RoundBrackets, DoubleQuotes, DatesToken)

  def parse(token:String):Seq[CandidateSplit] = {
    val splits = DictWord.splits(token)

    // give precedence to words coming from vocabulary
    if(splits.length == 0) {
      parsers.flatMap(_.splits(token))
    }
    else {
      splits
    }
  }
}

