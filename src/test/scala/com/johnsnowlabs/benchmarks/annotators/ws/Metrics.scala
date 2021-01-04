package com.johnsnowlabs.benchmarks.annotators.ws

case class Metrics(index: Int, precision: Double, recall: Double, fScore: Double)

case class AccuracyByParameter(nIter: Int, freqT: Int, ambT: Double, precision: Double, recall: Double, fScore: Double)