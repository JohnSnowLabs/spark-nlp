import unittest

import pytest
from pyspark.ml.param import Param, Params, TypeConverters

from sparknlp.internal import ParamsGettersSetters
from test.util import SparkSessionForTest


class ParamGettersSettersTestSpec(unittest.TestCase):
    spark = SparkSessionForTest.spark

    class MockAnnotator(ParamsGettersSetters):
        testStrParam = Param(
            Params._dummy(),
            "testStrParam",
            "Str Parameter for Test",
            typeConverter=TypeConverters.toString,
        )

        testFloatParam = Param(
            Params._dummy(),
            "testFloatParam",
            "Float Parameter for Test",
            typeConverter=TypeConverters.toFloat,
        )

        testListIntParam = Param(
            Params._dummy(),
            "testListIntParam",
            "List Int Parameter for Test",
            typeConverter=TypeConverters.toListInt,
        )

        testListBoolParam = Param(
            Params._dummy(),
            "testListBoolParam",
            "ListBool Parameter for Test",
            typeConverter=TypeConverters.toList,
        )

        testListFloatParam = Param(
            Params._dummy(),
            "testListFloatParam",
            "ListFloat Parameter for Test",
            typeConverter=TypeConverters.toListFloat,
        )

        testListNoTypeParam = Param(
            Params._dummy(), "testListNoTypeParam", "List Parameter without type"
        )

        param_vals = {}

        def _call_java(self, function_name, value):
            self.param_vals[function_name] = value

        def getOrDefault(self, paramName):
            param_name = f"set{paramName[0].upper()}{paramName[1:]}"
            return self.param_vals[param_name]

    annotator = MockAnnotator()

    def test_set_param_value(self):
        self.annotator.setTestStrParam("test")

    def test_get_param_value(self):
        self.annotator.setTestStrParam("test")
        assert self.annotator.getTestStrParam() == "test"

    def test_set_param_value_list(self):
        int_list = [1, 2, 3]
        self.annotator.setTestListIntParam(int_list)
        get_int_list = self.annotator.getTestListIntParam()
        assert type(get_int_list) != list
        for i1, i2 in zip(int_list, get_int_list):
            assert i1 == i2

        float_list = [0.1, 0.2, 0.3]
        self.annotator.setTestListFloatParam(float_list)
        get_float_list = self.annotator.getTestListFloatParam()
        assert type(get_float_list) != list
        for i1, i2 in zip(float_list, get_float_list):
            assert i1 == i2

    def test_set_param_custom_list(self):
        bool_list = [True, False, True]
        self.annotator.setTestListBoolParam(bool_list)
        get_bool_list = self.annotator.getTestListBoolParam()
        assert type(get_bool_list) != list
        for i1, i2 in zip(bool_list, get_bool_list):
            assert i1 == i2

    def test_set_param_custom_list_not_implemented(self):
        value = [self.annotator]
        with pytest.raises(ValueError):
            self.annotator.setTestListNoTypeParam(value)

    def test_set_empty_param_list(self):
        self.annotator.setTestListIntParam([])

        with pytest.raises(ValueError):
            self.annotator.setTestListNoTypeParam([])

    def test_type_conversion(self):
        with pytest.raises(TypeError):
            self.annotator.setTestStrParam(1)

        self.annotator.setTestFloatParam(5)
        get_val = self.annotator.getTestFloatParam()
        assert type(get_val) is float
        assert get_val == 5.0
