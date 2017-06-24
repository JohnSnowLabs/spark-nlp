package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.common.TaggedWord
import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach

import scala.collection.mutable.{Map => MMap}

/**
  * Created by saif on 24/06/17.
  */
case class SerializedPerceptronApproach(
                                       id: String,
                                       tags: List[String],
                                       wordBook: List[(String, String)],
                                       featuresWeight: Map[String, Map[String, Double]],
                                       lastIteration: Int
                                     ) extends SerializedAnnotatorApproach[PerceptronApproach] {
  override def deserialize: PerceptronApproach = {
    new PerceptronApproach(new AveragedPerceptron(
      tags.toArray,
      wordBook.map{ w: (String, String) => TaggedWord(w._1, w._2)}.toArray,
      MMap[String, MMap[String, Double]](featuresWeight.mapValues(c =>
        MMap[String, Double](c.toSeq:_*)).toSeq:_*),
      lastIteration
    ))
  }
}
object SerializedPerceptronApproach {
  val id: String = "perceptron"
}