package com.johnsnowlabs.nlp

import org.apache.spark.sql.{DataFrame, Dataset}

object SparkNLP {

  trait NLPBase {

    case class NLPBasic(
                          text: String,
                          tokens: Array[String],
                          normalized: Array[String],
                          lemmas: Array[String],
                          pos: Array[String]
                        )

    case class NLPAdvanced(
                            text: String,
                            tokens: Array[String],
                            normalized: Array[String],
                            spelled: Array[String],
                            stems: Array[String],
                            lemmas: Array[String],
                            pos: Array[String],
                            entities: Array[String]
                          )

    def basic(dataset: DataFrame, inputColumn: String): Dataset[NLPBasic]

    def basic(target: String): NLPBasic

    def basic(target: Array[String]): Array[NLPBasic]

    def advanced(dataset: DataFrame, inputColumn: String): Dataset[NLPAdvanced]

    def advanced(target: String): NLPBasic

    def advanced(target: Array[String]): Array[NLPBasic]

    def spellcheck(dataset: DataFrame, inputColumn: String): Dataset[String]

    def spellcheck(target: String): String

    def spellcheck(target: Array[String]): Array[String]

    def sentiment(dataset: DataFrame, inputColumn: String): Dataset[String]

    def sentiment(target: String): String

    def sentiment(target: Array[String]): Array[String]

  }

  object en extends NLPBase {
    override def basic(dataset: DataFrame, inputColumn: String): Dataset[en.NLPBasic] = ???

    override def basic(target: String): en.NLPBasic = ???

    override def basic(target: Array[String]): Array[en.NLPBasic] = ???

    override def advanced(dataset: DataFrame, inputColumn: String): Dataset[en.NLPAdvanced] = ???

    override def advanced(target: String): en.NLPBasic = ???

    override def advanced(target: Array[String]): Array[en.NLPBasic] = ???

    override def spellcheck(dataset: DataFrame, inputColumn: String): Dataset[String] = ???

    override def spellcheck(target: String): String = ???

    override def spellcheck(target: Array[String]): Array[String] = ???

    override def sentiment(dataset: DataFrame, inputColumn: String): Dataset[String] = ???

    override def sentiment(target: String): String = ???

    override def sentiment(target: Array[String]): Array[String] = ???
  }

}
