#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for the PromptAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class PromptAssembler(AnnotatorTransformer):
    """Assembles a sequence of messages into a single string using a template. These strings can then
    be used as prompts for large language models.

    This annotator expects an array of two-tuples as the type of the input column (one array of
    tuples per row). The first element of the tuples should be the role and the second element is
    the text of the message. Possible roles are "system", "user" and "assistant".

    An assistant header can be added to the end of the generated string by using
    ``setAddAssistant(True)``.

    At the moment, this annotator uses llama.cpp as a backend to parse and apply the templates.
    llama.cpp uses basic pattern matching to determine the type of the template, then applies a
    basic version of the template to the messages. This means that more advanced templates are not
    supported.

    For an extended example see the
    `example notebook <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/llama.cpp/PromptAssember_with_AutoGGUFModel.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCol
        Input column name
    outputCol
        Output column name
    chatTemplate
        Template used for the chat
    addAssistant
        Whether to add an assistant header to the end of the generated string

    Examples
    --------
    >>> from sparknlp.base import *
    >>> messages = [
    ...     [
    ...         ("system", "You are a helpful assistant."),
    ...         ("assistant", "Hello there, how can I help you?"),
    ...         ("user", "I need help with organizing my room."),
    ...     ]
    ... ]
    >>> df = spark.createDataFrame([messages]).toDF("messages")
    >>> template = (
    ...     "{{- bos_token }} {%- if custom_tools is defined %} {%- set tools = custom_tools %} {%- "
    ...     "endif %} {%- if not tools_in_user_message is defined %} {%- set tools_in_user_message = true %} {%- "
    ...     'endif %} {%- if not date_string is defined %} {%- set date_string = "26 Jul 2024" %} {%- endif %} '
    ...     "{%- if not tools is defined %} {%- set tools = none %} {%- endif %} {#- This block extracts the "
    ...     "system message, so we can slot it into the right place. #} {%- if messages[0]['role'] == 'system' %}"
    ...     " {%- set system_message = messages[0]['content']|trim %} {%- set messages = messages[1:] %} {%- else"
    ...     ' %} {%- set system_message = "" %} {%- endif %} {#- System message + builtin tools #} {{- '
    ...     '"<|start_header_id|>system<|end_header_id|>\\n\\n" }} {%- if builtin_tools is defined or tools is '
    ...     'not none %} {{- "Environment: ipython\\n" }} {%- endif %} {%- if builtin_tools is defined %} {{- '
    ...     '"Tools: " + builtin_tools | reject(\\'equalto\', \\'code_interpreter\\') | join(", ") + "\\n\\n"}} '
    ...     '{%- endif %} {{- "Cutting Knowledge Date: December 2023\\n" }} {{- "Today Date: " + date_string '
    ...     '+ "\\n\\n" }} {%- if tools is not none and not tools_in_user_message %} {{- "You have access to '
    ...     'the following functions. To call a function, please respond with JSON for a function call." }} {{- '
    ...     '\\'Respond in the format {"name": function name, "parameters": dictionary of argument name and its'
    ...     ' value}.\\' }} {{- "Do not use variables.\\n\\n" }} {%- for t in tools %} {{- t | tojson(indent=4) '
    ...     '}} {{- "\\n\\n" }} {%- endfor %} {%- endif %} {{- system_message }} {{- "<|eot_id|>" }} {#- '
    ...     "Custom tools are passed in a user message with some extra guidance #} {%- if tools_in_user_message "
    ...     "and not tools is none %} {#- Extract the first user message so we can plug it in here #} {%- if "
    ...     "messages | length != 0 %} {%- set first_user_message = messages[0]['content']|trim %} {%- set "
    ...     'messages = messages[1:] %} {%- else %} {{- raise_exception("Cannot put tools in the first user '
    ...     "message when there's no first user message!\\") }} {%- endif %} {{- "
    ...     "'<|start_header_id|>user<|end_header_id|>\\n\\n' -}} {{- \\"Given the following functions, please "
    ...     'respond with a JSON for a function call " }} {{- "with its proper arguments that best answers the '
    ...     'given prompt.\\n\\n" }} {{- \\'Respond in the format {"name": function name, "parameters": '
    ...     'dictionary of argument name and its value}.\\' }} {{- "Do not use variables.\\n\\n" }} {%- for t in '
    ...     'tools %} {{- t | tojson(indent=4) }} {{- "\\n\\n" }} {%- endfor %} {{- first_user_message + '
    ...     "\\"<|eot_id|>\\"}} {%- endif %} {%- for message in messages %} {%- if not (message.role == 'ipython' "
    ...     "or message.role == 'tool' or 'tool_calls' in message) %} {{- '<|start_header_id|>' + message['role']"
    ...     " + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }} {%- elif 'tool_calls' in "
    ...     'message %} {%- if not message.tool_calls|length == 1 %} {{- raise_exception("This model only '
    ...     'supports single tool-calls at once!") }} {%- endif %} {%- set tool_call = message.tool_calls[0]'
    ...     ".function %} {%- if builtin_tools is defined and tool_call.name in builtin_tools %} {{- "
    ...     "'<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- \\"<|python_tag|>\\" + tool_call.name + "
    ...     '".call(" }} {%- for arg_name, arg_val in tool_call.arguments | items %} {{- arg_name + \\'="\\' + '
    ...     'arg_val + \\'"\\' }} {%- if not loop.last %} {{- ", " }} {%- endif %} {%- endfor %} {{- ")" }} {%- '
    ...     "else %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- '{\\"name\": \\"' + "
    ...     'tool_call.name + \\'", \\' }} {{- \\'"parameters": \\' }} {{- tool_call.arguments | tojson }} {{- "}" '
    ...     "}} {%- endif %} {%- if builtin_tools is defined %} {#- This means we're in ipython mode #} {{- "
    ...     '"<|eom_id|>" }} {%- else %} {{- "<|eot_id|>" }} {%- endif %} {%- elif message.role == "tool" '
    ...     'or message.role == "ipython" %} {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }} {%- '
    ...     "if message.content is mapping or message.content is iterable %} {{- message.content | tojson }} {%- "
    ...     'else %} {{- message.content }} {%- endif %} {{- "<|eot_id|>" }} {%- endif %} {%- endfor %} {%- if '
    ...     "add_generation_prompt %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }} {%- endif %} "
    ... )
    >>> prompt_assembler = (
    ...     PromptAssembler()
    ...     .setInputCol("messages")
    ...     .setOutputCol("prompt")
    ...     .setChatTemplate(template)
    ... )
    >>> prompt_assembler.transform(df).select("prompt.result").show(truncate=False)
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                                                                                                                                                      |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\nHello there, how can I help you?<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nI need help with organizing my room.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n]|
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    outputAnnotatorType = AnnotatorType.DOCUMENT

    inputCol = Param(
        Params._dummy(),
        "inputCol",
        "input column name",
        typeConverter=TypeConverters.toString,
    )
    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name",
        typeConverter=TypeConverters.toString,
    )
    chatTemplate = Param(
        Params._dummy(),
        "chatTemplate",
        "Template used for the chat",
        typeConverter=TypeConverters.toString,
    )
    addAssistant = Param(
        Params._dummy(),
        "addAssistant",
        "Whether to add an assistant header to the end of the generated string",
        typeConverter=TypeConverters.toBoolean,
    )
    name = "PromptAssembler"

    @keyword_only
    def __init__(self):
        super(PromptAssembler, self).__init__(
            classname="com.johnsnowlabs.nlp.PromptAssembler"
        )
        self._setDefault(outputCol="prompt", addAssistant=True)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the input column
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)

    def setChatTemplate(self, value):
        """Sets the chat template.

        Parameters
        ----------
        value : str
            Template used for the chat
        """
        return self._set(chatTemplate=value)

    def setAddAssistant(self, value):
        """Sets whether to add an assistant header to the end of the generated string.

        Parameters
        ----------
        value : bool
            Whether to add an assistant header to the end of the generated string
        """
        return self._set(addAssistant=value)
