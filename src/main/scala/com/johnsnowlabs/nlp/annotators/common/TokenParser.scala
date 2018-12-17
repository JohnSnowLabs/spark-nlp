package com.johnsnowlabs.nlp.annotators.common

trait PreprocessingParser {
  def separate(token:String): String
}


class SuffixedToken(suffixes:Array[String]) extends PreprocessingParser {

  def belongs(token: String): Option[String] =
    suffixes.find(token.endsWith)


  override def separate(token:String): String = {
    belongs(token).map { suffix  =>
      s"""${separate(token.dropRight(suffix.length))} $suffix"""
    }.getOrElse(token)
  }

}

object SuffixedToken {
  def apply(suffixes:Array[String]) = new SuffixedToken(suffixes)
}


class PrefixedToken(prefixes:Array[String]) extends PreprocessingParser {

  private def parse(token:String)  =
    (token.head.toString, token.tail)

  def belongs(token: String): Boolean =
    if(token.length > 1)
      prefixes.map(token.head.toString.equals).reduce(_ || _)
    else
      false

  override def separate(token:String): String = {
    if (belongs(token))
      s"""${token.head} ${separate(token.tail)}"""
    else
      token
  }
}

object PrefixedToken {
  def apply(prefixes:Array[String]) = new PrefixedToken(prefixes)
}


class InfixToken(tokens:Array[String]) extends PreprocessingParser {

  private def parse(token:String)  =
    (token.head.toString, token.tail)

  def belongs(token: String): Boolean = {
    if(token.length > 2) {
      val insideChunk = token.tail.dropRight(1)
      tokens.exists(insideChunk.contains)
    }else{
      false
    }
  }

  override def separate(token:String): String = {
    var result = token
    if (belongs(token)) {
      tokens.foreach{ t =>
        result = token.replaceAll(t, s" $t ")
      }
    }
    result
  }
}

object InfixToken {
  def apply(prefixes:Array[String]) = new InfixToken(prefixes)
}