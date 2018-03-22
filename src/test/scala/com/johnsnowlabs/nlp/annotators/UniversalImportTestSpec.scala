package com.johnsnowlabs.nlp.annotators

import org.scalatest.FlatSpec

class UniversalImportTestSpec extends FlatSpec {

  "A SentenceDetector" should "be imported automatically when accessing annotator pseudo package" in {
    /** Now you can access all annotators by using this import here */
    import com.johnsnowlabs.nlp.annotator._

    /** For example */
    val sentenceDetector = new SentenceDetector()
    val SSD_PATH = "./tst_shortcut_sd"
    sentenceDetector.write.overwrite().save(SSD_PATH)
    SentenceDetector.read.load(SSD_PATH)
  }

}
