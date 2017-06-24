package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.annotators.common.AnnotatorApproach

/**
  * Created by Saif Addin on 5/5/2017.
  */
trait SBDApproach extends AnnotatorApproach {

  val description: String

  private var initialized: Boolean = false
  private var content: Option[String] = None

  private[sbd] def overrideContent(target: String): this.type = {
    content = Some(target)
    this
  }

  def setContent(target: String): this.type = {
    if (!initialized) {
      content = Some(target)
      initialized = true
    }
    this
  }

  protected def updateContent(target: String): this.type = {
    content = Some(target)
    this
  }

  protected def getContent: Option[String] = content

  def prepare: SBDApproach

  def extract: Array[Sentence]

}