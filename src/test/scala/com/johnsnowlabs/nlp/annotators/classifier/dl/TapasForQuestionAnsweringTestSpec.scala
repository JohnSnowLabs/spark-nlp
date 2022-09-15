package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class TapasForQuestionAnsweringTestSpec extends AnyFlatSpec {

  "TapasForQuestionAnswering" should "load saved model" taggedAs SlowTest ignore {
    TapasForQuestionAnswering
      .loadSavedModel("/tmp/tapas_tf", ResourceHelper.spark)
      .save("/models/sparknlp/tapas")
  }

  "TapasForQuestionAnswering" should "prepare inputs" in {

  }
}
