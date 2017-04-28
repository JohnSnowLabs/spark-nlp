package com.jsl.nlp

import org.apache.spark.sql.Row

/**
  * Created by saif on 28/04/17.
  */
trait ExtractsFromRow {

  /**
    *
    * @param row Row to be validated
    */
  def validate(row: Row): Unit

}
