package com.johnsnowlabs.nlp.annotators.common

import org.apache.spark.ml.param._

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._


class IntStringMapParam(parent: Params, name: String, doc: String, isValid: Map[String, Int] => Boolean)
  extends Param[Map[String, Int]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, (_: Map[String, Int]) => true)

  /** Creates a param pair with a `java.util.List` of values (for Java and Python). */
  def w(value: java.util.HashMap[String, Int]): ParamPair[Map[String, Int]] = w(value.asScala.toMap)

  override def jsonEncode(value: Map[String, Int]): String = {
    import org.json4s.JsonDSL._
    compact(render(value.toSeq))
  }

  override def jsonDecode(json: String): Map[String, Int] = {
    implicit val formats = DefaultFormats
    parse(json).extract[Seq[(String, Int)]].toMap
  }

}
