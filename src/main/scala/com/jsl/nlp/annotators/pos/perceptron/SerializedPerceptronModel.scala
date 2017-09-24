package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.common.TaggedWord
import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent

import scala.collection.mutable.{Map => MMap}

/**
  * Created by saif on 24/06/17.
  */

/**
  * Serialized representation of [[PerceptronApproach]]
  * Converting mutable types and arrays into serializable lists
  * @param tags unique set of POS-tags stored
  * @param wordBook book that contains non-ambiguous tags
  * @param featuresWeight features that contain a context and frequency of appearance based on training
  * @param lastIteration last iteration run on training, useful for weighting
  */
case class SerializedPerceptronModel(
                                       tags: List[String],
                                       wordBook: List[(String, String)],
                                       featuresWeight: Map[String, Map[String, Double]],
                                       lastIteration: Int
                                     ) extends SerializedAnnotatorComponent[AveragedPerceptron] {

  /** Puts back the read content into the original form of [[com.jsl.nlp.annotators.pos.perceptron.PerceptronApproach]]
    * and its contained [[AveragedPerceptron]]
    */
  override def deserialize: AveragedPerceptron = {
    new AveragedPerceptron(
      tags.toArray,
      wordBook.map{ w: (String, String) => TaggedWord(w._1, w._2)}.toArray,
      MMap[String, MMap[String, Double]](featuresWeight.mapValues(c =>
        MMap[String, Double](c.toSeq:_*)).toSeq:_*),
      lastIteration
    )
  }
}