#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains utility classes for ParamsGettersSetters."""

import re

from pyspark.ml.linalg import DenseVector, Matrix
from pyspark.ml.param import Params

from sparknlp.internal.converters import list_to_java_array
from sparknlp.internal.doc_string_generator import DocStringGenerator


class ParamsGettersSetters(Params):
    """Class that generates setter and getter functions for parameters defined in an annotator class.

    The setter function will attempt to convert the provided value and call the Scala side setter function.
    """

    def __init__(self):
        super(ParamsGettersSetters, self).__init__()
        for param in self.params:
            param_name = param.name
            getter_function_name = "get" + re.sub(
                r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name
            )
            setter_function_name = "set" + re.sub(
                r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name
            )

            # Generates getter and setter only if not exists
            if not hasattr(self, getter_function_name):
                getter_function = self._get_getter_function(param_name)
                getter_function.__doc__ = DocStringGenerator.generate_numpydoc_getter(
                    param.doc
                )

                setattr(self, getter_function_name, getter_function)

            if not hasattr(self, setter_function_name):
                setter_function = self._get_setter_java(
                    setter_function_name, param
                )
                setter_function.__doc__ = DocStringGenerator.generate_numpydoc_setter(
                    param
                )

                setattr(self, setter_function_name, setter_function)

    def _get_getter_function(self, paramName):
        """Returns the getter function of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter

        Returns
        -------
        Getter Function of the parameter name
        """

        def getter():
            try:
                return self.getOrDefault(paramName)
            except KeyError:
                return None

        return getter

    def _get_setter_function(self, paramName):
        """Returns the setter function of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter

        Returns
        -------
        Setter Function of the parameter name
        """

        def setter(v):
            self.set(self.getParam(paramName), v)
            return self

        return setter

    def _get_setter_java(self, setter_name, param):
        """Returns a setter function, which calls the equivalent on the java side.

        Parameters
        ----------
        setter_name : str
            Name of the setter of parameter
        type_converter
            TypeConvert used for the parameter

        Returns
        -------
        Setter Function for the java side setter
        """
        type_converter = param.typeConverter

        def setter(value):
            """
            Calls the setter on the java side, using the provided value.

            If the value is a list, the type will be converted to be compatible with Scala Arrays.

            Parameters
            ----------
            value
                The provided value to set

            Returns
            -------
            self
            """
            value = type_converter(value)

            if type(value) is list:
                value = list_to_java_array(value, type_converter)

            if type(value) is DenseVector or type(value) is Matrix:
                self.set(param, value)  # Set directly
            else:
                self._call_java(setter_name, value)
            return self

        return setter
