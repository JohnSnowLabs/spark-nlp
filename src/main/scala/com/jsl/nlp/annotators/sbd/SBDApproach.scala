package com.jsl.nlp.annotators.sbd

/**
  * Created by Saif Addin on 5/5/2017.
  */
trait SBDApproach {

  val description: String

  def prepare: SBDApproach

  def extract: Array[Sentence]

}