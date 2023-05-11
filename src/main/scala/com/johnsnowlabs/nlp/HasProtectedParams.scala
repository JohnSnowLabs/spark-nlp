package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{Param, Params}

/** Enables a class to protect a parameter, which means that it can only be set once.
  *
  * This trait will enable a implicit conversion from Param to ProtectedParam. In addition, the
  * new set for ProtectedParam will then check, whether or not the value was already set. If so,
  * then a warning will be output and the value will not be set again.
  */
trait HasProtectedParams {
  this: Params =>
  implicit class ProtectedParam[T](private val param: Param[T])
      extends Param[T](param.parent, param.name, param.doc, param.isValid) {

    var isProtected = false

    /** Sets this parameter to be protected, which means that it can only be set once.
      *
      * Default values do not count as a set value and can be overridden.
      *
      * @return
      *   This object
      */
    def setProtected(): this.type = {
      isProtected = true
      this
    }

    def toParam: Param[T] = this.asInstanceOf[Param[T]]
  }

  /** Sets the value for a protected Param.
    *
    * If the parameter was already set, it will not be set again. Default values do not count as a
    * set value and can be overridden.
    *
    * @param param
    *   Protected parameter to set
    * @param value
    *   Value for the parameter
    * @tparam T
    *   Type of the parameter
    * @return
    *   This object
    */
  def set[T](param: ProtectedParam[T], value: T): this.type = {
    if (param.isProtected && get(param).isDefined)
      println(
        s"Warning: The parameter ${param.name} is protected and can only be set once." +
          " For a pretrained model, this was done during the initialization process." +
          " If you are trying to train your own model, please check the documentation.")
    else
      set(param.toParam, value)
    this
  }
}
