package com.jsl.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap}

/**
  * Created by Saif Addin on 5/16/2017.
  */
class AveragedPerceptron {

  val featuresWeight: MMap[String, MMap[String, Double]] = MMap()
  var classes: Set[String] = Set()

  var nIteration: Int = 0
  private val totals: MMap[(String, String), Double] = MMap().withDefaultValue(0.0)
  private val timestamps: MMap[(String, String), Double] = MMap().withDefaultValue(0.0)

  def predict(features: Map[String, Int]): String = {
    /**
      * scores are used for feature scores, which are all by default 0
      * if a feature has a relevant score, look for all its possible tags and their scores
      * multiply their weights per the times they appear
      * Return highest tag by score
      *
      */
    val scores: MMap[String, Double] = MMap().withDefaultValue(0.0)
    features
      .filter{case (feature, value) => featuresWeight.contains(feature) && value != 0}
      .map{case (feature, value ) => (featuresWeight(feature), value)}
      .foreach{case (tagsWeight, value) =>
        tagsWeight.foreach{case (tag, weight) =>
          scores(tag) += (value * weight)
        }
      }
    classes.maxBy((tag) => (scores(tag),tag))
  }

  /**
    * once a model was trained, average its weights more in the first iterations
    */
  def averageWeights(): Unit = {
    featuresWeight.foreach{case (feature, weights) =>
      val newWeights: MMap[String, Double] = MMap()
      weights.foreach{case (tag, weight) =>
        val param = (feature, tag)
        val total = (totals(param) + (nIteration - timestamps(param))) * weight
        newWeights(tag) = total / nIteration.toDouble
        featuresWeight(feature) = newWeights
      }
    }
  }

  def update(truth: String, guess: String, features: Map[String, Int]): Unit = {
    def updateFeature(tag: String, feature: String, weight: Double, value: Double) = {
      val param = (feature, tag)
      /**
        * update totals and timestamps
        */
      totals(param) += ((nIteration - timestamps(param)) * weight)
      timestamps(param) = nIteration
      /**
        * update weights
        */
      featuresWeight(feature)(tag) = weight + value
    }
    nIteration += 1
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