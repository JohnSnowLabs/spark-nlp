package com.johnsnowlabs.nlp

import org.apache.spark.sql.DataFrame

object EasyNLP {

  private case class NLPBasics(
                                text: String,
                                tokens: Array[String],
                                normalized: Array[String],
                                lemmas: Array[String],
                                pos: Array[String]
                              )

  private case class NLPAdvancedd(
                                   text: String,
                                   tokens: Array[String],
                                   normalized: Array[String],
                                   lemmas: Array[String],
                                   stems: Array[String],
                                   spelled: Array[String],
                                   pos: Array[String],
                                   entities: Array[String]
                                 )

  def basic(dataset: DataFrame, inputColumn: String): DataFrame = {
    ???
  }

  def basic(target: String): NLPBasics = {
    ???
  }

  def basic(target: Array[String]): Array[NLPBasics] = {
    ???
  }

  def advanced(dataset: DataFrame, inputColumn: String): DataFrame = {
    ???
  }

  def advanced(target: String): NLPBasics = {
    ???
  }

  def advanced(target: Array[String]): Array[NLPBasics] = {
    ???
  }

  def spellcheck(dataset: DataFrame, inputColumn: String): DataFrame = {
    ???
  }

  def spellcheck(target: String): String = {
    ???
  }

  def spellcheck(target: Array[String]): Array[String] = {
    ???
  }

  def sentiment(dataset: DataFrame, inputColumn: String): DataFrame = {
    ???
  }

  def sentiment(target: String): String = {
    ???
  }

  def sentiment(target: Array[String]): Array[String] = {
    ???
  }

}
