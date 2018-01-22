package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model

/**
  * Created by jose on 21/01/18.
  * This class allows for model evaluation happening on distributed Spark collections
  */
trait DatasetAnnotatorModel[M <: Model[M]] extends BaseAnnotatorModel[M] {

}
