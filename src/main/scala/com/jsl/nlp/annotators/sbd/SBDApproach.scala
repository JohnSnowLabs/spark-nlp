package com.jsl.nlp.annotators.sbd

/**
  * Created by Saif Addin on 5/5/2017.
  */
trait SBDApproach {

  val description: String

  def prepare(text: String): RawContent

  /**
    * @return
    */
  def clean(rawContent: RawContent): ReadyContent

  /**
    * Goes through all sentences and records substring beginning and end
    * May think a more functional way? perhaps a foldRight with vector (i,lastChar)?
    * @return
    */
  def extract(readyContent: ReadyContent): Array[Sentence]

}