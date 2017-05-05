package com.jsl.nlp.annotators.sbd

/**
  * Created by Saif Addin on 5/5/2017.
  */
trait RawContent {
  val approach: SBDApproach
  final def clean: ReadyContent = approach.clean(this)
}

trait ReadyContent {
  val approach: SBDApproach
  final def extract: Array[Sentence] = approach.extract(this)
}
