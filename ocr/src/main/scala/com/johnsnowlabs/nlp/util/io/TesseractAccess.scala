package com.johnsnowlabs.nlp.util.io

import net.sourceforge.tess4j.Tesseract
class TesseractAccess extends Tesseract {

  def initialize() = {
    this.init()
  }
}
