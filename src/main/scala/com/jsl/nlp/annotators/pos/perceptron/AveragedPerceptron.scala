package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.pos.POSModel

import scala.collection.mutable.{Map => MMap}

/**
  * Created by Saif Addin on 5/16/2017.
  */
class AveragedPerceptron(
                          val taggedWordBook: List[TaggedWord],
                          tags: List[String],
                          initialWeights: MMap[String, MMap[String, Double]],
                          lastIteration: Int = 0
                        ) extends POSModel {

  private var updateIteration: Int = lastIteration
  private val featuresWeight: MMap[String, MMap[String, Double]] = initialWeights
  private val totals: MMap[(String, String), Double] = MMap().withDefaultValue(0.0)
  private val timestamps: MMap[(String, String), Double] = MMap().withDefaultValue(0.0)

  override def predict(features: Map[String, Int]): String = {
    /**
      * scores are used for feature scores, which are all by default 0
      * if a feature has a relevant score, look for all its possible tags and their scores
      * multiply their weights per the times they appear
      * Return highest tag by score
      *
      */
    val scoresByTag = features
      .filter{case (feature, value) => featuresWeight.contains(feature) && value != 0}
      .map{case (feature, value ) => (featuresWeight(feature), value)}
      .flatMap{case (tagsWeight, value) => tagsWeight.map{ case (tag, weight) => (tag, value * weight)}
      }
    /**
      * ToDo: Watch it here. Because of missing training corpus, default values are made to make tests pass
      */
    tags.maxBy{ tag => scoresByTag.withDefaultValue(0.0)(tag)}
  }

  /**
    * Training level operation
    * once a model was trained, average its weights more in the first iterations
    */
  def averagedModel: AveragedPerceptron = {
    new AveragedPerceptron(
      taggedWordBook,
      tags,
      featuresWeight.map { case (feature, weights) =>
        (feature,
          weights.map { case (tag, weight) =>
            val param = (feature, tag)
            val total = (totals(param) + (updateIteration - timestamps(param))) * weight
            (tag, total / updateIteration.toDouble)
          }
        )
      },
      updateIteration
    )
  }

  def getUpdateIterations: Int = updateIteration
  def getTags: List[String] = tags
  /**
    * This is model learning tweaking during training, in-place
    * Uses mutable collections since this runs per word, not per iteration
    * Hence, performance is needed, without risk as long as this is a
    * non parallel training running outside spark
    * @return
    */
  def update(truth: String, guess: String, features: Map[String, Int]): Unit = {
    def updateFeature(tag: String, feature: String, weight: Double, value: Double) = {
      val param = (feature, tag)
      /**
        * update totals and timestamps
        */
      totals(param) += ((updateIteration - timestamps(param)) * weight)
      timestamps(param) = updateIteration
      /**
        * update weights
        */
      featuresWeight(feature)(tag) = weight + value
    }
    updateIteration += 1
    /**
      * if prediction was wrong, take all features and for each feature get feature's current tags and their weights
      * congratulate if success and punish for wrong in weight
      */
    if (truth != guess) {
      features.foreach{case (feature, _) =>
        val weights = featuresWeight.getOrElseUpdate(feature, MMap())
        updateFeature(truth, feature, weights.getOrElse(truth, 0.0), 1.0)
        updateFeature(guess, feature, weights.getOrElse(guess, 0.0), -1.0)
      }
    }
  }

}