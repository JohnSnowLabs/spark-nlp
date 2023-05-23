from pyspark.ml.param import TypeConverters


def type_converter_to_type_string(type_converter):
    if type_converter == TypeConverters.toListInt:
        return "List[int]"
    elif type_converter == TypeConverters.toList:
        return "List"
    elif type_converter == TypeConverters.toListFloat:
        return "List[float]"
    elif type_converter == TypeConverters.toListString:
        return "List[str]"
    elif type_converter == TypeConverters.toVector:
        return "Vector"
    elif type_converter == TypeConverters.toMatrix:
        return "Matrix"
    elif type_converter == TypeConverters.toFloat:
        return "float"
    elif type_converter == TypeConverters.toInt:
        return "int"
    elif type_converter == TypeConverters.toString:
        return "str"
    elif type_converter == TypeConverters.toBoolean:
        return "bool"
    else:
        return None


class DocStringGenerator:
    @staticmethod
    def generate_numpydoc_getter(description):
        description = description[0].lower() + description[1:]
        doc_string = f"Gets {description}."
        return doc_string

    @staticmethod
    def generate_numpydoc_setter(param, default_value=None):
        description = param.doc
        description = description[0].lower() + description[1:]
        description = (
            description
            if not default_value
            else description + f", by default {default_value}"
        )
        param_type = type_converter_to_type_string(param.typeConverter)
        param_type = f" : {param_type}" if param_type else ""
        doc_string = (
            f"Sets {description}.\n"
            f"\n"
            f"Parameters\n"
            f"----------\n"
            f"value{param_type}\n"
            f"    {description}"
        )  # TODO: Default value
        return doc_string
