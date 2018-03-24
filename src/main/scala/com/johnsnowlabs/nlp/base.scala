package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.DefaultParamsReadable

object base {

  type DocumentAssembler = com.johnsnowlabs.nlp.DocumentAssembler
  object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]

  type TokenAssembler = com.johnsnowlabs.nlp.TokenAssembler
  object TokenAssembler extends DefaultParamsReadable[TokenAssembler]

  type Finisher = com.johnsnowlabs.nlp.Finisher
  object Finisher extends DefaultParamsReadable[Finisher]

}
