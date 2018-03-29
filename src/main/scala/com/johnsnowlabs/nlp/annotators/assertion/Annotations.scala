package com.johnsnowlabs.nlp.annotators.assertion

/**
  * Created by jose on 19/03/18.
  */
case class Datapoint(sentence: String, target: String, label: String, start:Int, end:Int)
