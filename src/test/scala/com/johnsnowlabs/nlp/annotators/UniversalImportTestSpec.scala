package com.johnsnowlabs.nlp.annotators

import org.scalatest.FlatSpec

class UniversalImportTestSpec extends FlatSpec {

  "A SentenceDetector" should "be imported automatically when accessing annotator pseudo package" in {
    import com.johnsnowlabs.nlp.annotator._
    val sd = new SentenceDetector()
    val SSD_PATH = "./tst_shortcut_sd"
    sd.write.save(SSD_PATH)
    SentenceDetector.read.load(SSD_PATH)
  }

}
