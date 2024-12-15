{%- capture title -%}
PromptAssembler
{%- endcapture -%}

{%- capture description -%}
Assembles a sequence of messages into a single string using a template. These strings can then
be used as prompts for large language models.

This annotator expects an array of two-tuples as the type of the input column (one array of
tuples per row). The first element of the tuples should be the role and the second element is
the text of the message. Possible roles are "system", "user" and "assistant".

An assistant header can be added to the end of the generated string by using
`setAddAssistant(true)`.

At the moment, this annotator uses llama.cpp as a backend to parse and apply the templates.
llama.cpp uses basic pattern matching to determine the type of the template, then applies a
basic version of the template to the messages. This means that more advanced templates are not
supported.

For an extended example see the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/llama.cpp/PromptAssember_with_AutoGGUFModel.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import *

messages = [
    [
        ("system", "You are a helpful assistant."),
        ("assistant", "Hello there, how can I help you?"),
        ("user", "I need help with organizing my room."),
    ]
]
df = spark.createDataFrame([messages]).toDF("messages")

{% raw %}
# llama3.1
template = (
    "{{- bos_token }} {%- if custom_tools is defined %} {%- set tools = custom_tools %} {%- "
    "endif %} {%- if not tools_in_user_message is defined %} {%- set tools_in_user_message = true %} {%- "
    'endif %} {%- if not date_string is defined %} {%- set date_string = "26 Jul 2024" %} {%- endif %} '
    "{%- if not tools is defined %} {%- set tools = none %} {%- endif %} {#- This block extracts the "
    "system message, so we can slot it into the right place. #} {%- if messages[0]['role'] == 'system' %}"
    " {%- set system_message = messages[0]['content']|trim %} {%- set messages = messages[1:] %} {%- else"
    ' %} {%- set system_message = "" %} {%- endif %} {#- System message + builtin tools #} {{- '
    '"<|start_header_id|>system<|end_header_id|>\\n\n" }} {%- if builtin_tools is defined or tools is '
    'not none %} {{- "Environment: ipython\\n" }} {%- endif %} {%- if builtin_tools is defined %} {{- '
    '"Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\n"}} '
    '{%- endif %} {{- "Cutting Knowledge Date: December 2023\\n" }} {{- "Today Date: " + date_string '
    '+ "\\n\n" }} {%- if tools is not none and not tools_in_user_message %} {{- "You have access to '
    'the following functions. To call a function, please respond with JSON for a function call." }} {{- '
    '\'Respond in the format {"name": function name, "parameters": dictionary of argument name and its'
    ' value}.\' }} {{- "Do not use variables.\\n\n" }} {%- for t in tools %} {{- t | tojson(indent=4) '
    '}} {{- "\\n\n" }} {%- endfor %} {%- endif %} {{- system_message }} {{- "<|eot_id|>" }} {#- '
    "Custom tools are passed in a user message with some extra guidance #} {%- if tools_in_user_message "
    "and not tools is none %} {#- Extract the first user message so we can plug it in here #} {%- if "
    "messages | length != 0 %} {%- set first_user_message = messages[0]['content']|trim %} {%- set "
    'messages = messages[1:] %} {%- else %} {{- raise_exception("Cannot put tools in the first user '
    "message when there's no first user message!\") }} {%- endif %} {{- "
    "'<|start_header_id|>user<|end_header_id|>\\n\n' -}} {{- \"Given the following functions, please "
    'respond with a JSON for a function call " }} {{- "with its proper arguments that best answers the '
    'given prompt.\\n\n" }} {{- \'Respond in the format {"name": function name, "parameters": '
    'dictionary of argument name and its value}.\' }} {{- "Do not use variables.\\n\n" }} {%- for t in '
    'tools %} {{- t | tojson(indent=4) }} {{- "\\n\n" }} {%- endfor %} {{- first_user_message + '
    "\"<|eot_id|>\"}} {%- endif %} {%- for message in messages %} {%- if not (message.role == 'ipython' "
    "or message.role == 'tool' or 'tool_calls' in message) %} {{- '<|start_header_id|>' + message['role']"
    " + '<|end_header_id|>\\n\n'+ message['content'] | trim + '<|eot_id|>' }} {%- elif 'tool_calls' in "
    'message %} {%- if not message.tool_calls|length == 1 %} {{- raise_exception("This model only '
    'supports single tool-calls at once!") }} {%- endif %} {%- set tool_call = message.tool_calls[0]'
    ".function %} {%- if builtin_tools is defined and tool_call.name in builtin_tools %} {{- "
    "'<|start_header_id|>assistant<|end_header_id|>\\n\n' -}} {{- \"<|python_tag|>\" + tool_call.name + "
    '".call(" }} {%- for arg_name, arg_val in tool_call.arguments | items %} {{- arg_name + \'="\' + '
    'arg_val + \'"\' }} {%- if not loop.last %} {{- ", " }} {%- endif %} {%- endfor %} {{- ")" }} {%- '
    "else %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\n' -}} {{- '{\"name\": \"' + "
    'tool_call.name + \'", \' }} {{- \'"parameters": \' }} {{- tool_call.arguments | tojson }} {{- "}" '
    "}} {%- endif %} {%- if builtin_tools is defined %} {#- This means we're in ipython mode #} {{- "
    '"<|eom_id|>" }} {%- else %} {{- "<|eot_id|>" }} {%- endif %} {%- elif message.role == "tool" '
    'or message.role == "ipython" %} {{- "<|start_header_id|>ipython<|end_header_id|>\\n\n" }} {%- '
    "if message.content is mapping or message.content is iterable %} {{- message.content | tojson }} {%- "
    'else %} {{- message.content }} {%- endif %} {{- "<|eot_id|>" }} {%- endif %} {%- endfor %} {%- if '
    "add_generation_prompt %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\n' }} {%- endif %} "
)
{% endraw %}

prompt_assembler = (
    PromptAssembler()
    .setInputCol("messages")
    .setOutputCol("prompt")
    .setChatTemplate(template)
)

prompt_assembler.transform(df).select("prompt.result").show(truncate=False)
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello there, how can I help you?<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI need help with organizing my room.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
// Batches (whole conversations) of arrays of messages
val data: Seq[Seq[(String, String)]] = Seq(
  Seq(
    ("system", "You are a helpful assistant."),
    ("assistant", "Hello there, how can I help you?"),
    ("user", "I need help with organizing my room.")))

val dataDF = data.toDF("messages")

{% raw %}
// llama3.1
val template =
  "{{- bos_token }} {%- if custom_tools is defined %} {%- set tools = custom_tools %} {%- " +
    "endif %} {%- if not tools_in_user_message is defined %} {%- set tools_in_user_message = true %} {%- " +
    "endif %} {%- if not date_string is defined %} {%- set date_string = \"26 Jul 2024\" %} {%- endif %} " +
    "{%- if not tools is defined %} {%- set tools = none %} {%- endif %} {#- This block extracts the " +
    "system message, so we can slot it into the right place. #} {%- if messages[0]['role'] == 'system' %}" +
    " {%- set system_message = messages[0]['content']|trim %} {%- set messages = messages[1:] %} {%- else" +
    " %} {%- set system_message = \"\" %} {%- endif %} {#- System message + builtin tools #} {{- " +
    "\"<|start_header_id|>system<|end_header_id|>\\n\\n\" }} {%- if builtin_tools is defined or tools is " +
    "not none %} {{- \"Environment: ipython\\n\" }} {%- endif %} {%- if builtin_tools is defined %} {{- " +
    "\"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}} " +
    "{%- endif %} {{- \"Cutting Knowledge Date: December 2023\\n\" }} {{- \"Today Date: \" + date_string " +
    "+ \"\\n\\n\" }} {%- if tools is not none and not tools_in_user_message %} {{- \"You have access to " +
    "the following functions. To call a function, please respond with JSON for a function call.\" }} {{- " +
    "'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its" +
    " value}.' }} {{- \"Do not use variables.\\n\\n\" }} {%- for t in tools %} {{- t | tojson(indent=4) " +
    "}} {{- \"\\n\\n\" }} {%- endfor %} {%- endif %} {{- system_message }} {{- \"<|eot_id|>\" }} {#- " +
    "Custom tools are passed in a user message with some extra guidance #} {%- if tools_in_user_message " +
    "and not tools is none %} {#- Extract the first user message so we can plug it in here #} {%- if " +
    "messages | length != 0 %} {%- set first_user_message = messages[0]['content']|trim %} {%- set " +
    "messages = messages[1:] %} {%- else %} {{- raise_exception(\"Cannot put tools in the first user " +
    "message when there's no first user message!\") }} {%- endif %} {{- " +
    "'<|start_header_id|>user<|end_header_id|>\\n\\n' -}} {{- \"Given the following functions, please " +
    "respond with a JSON for a function call \" }} {{- \"with its proper arguments that best answers the " +
    "given prompt.\\n\\n\" }} {{- 'Respond in the format {\"name\": function name, \"parameters\": " +
    "dictionary of argument name and its value}.' }} {{- \"Do not use variables.\\n\\n\" }} {%- for t in " +
    "tools %} {{- t | tojson(indent=4) }} {{- \"\\n\\n\" }} {%- endfor %} {{- first_user_message + " +
    "\"<|eot_id|>\"}} {%- endif %} {%- for message in messages %} {%- if not (message.role == 'ipython' " +
    "or message.role == 'tool' or 'tool_calls' in message) %} {{- '<|start_header_id|>' + message['role']" +
    " + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }} {%- elif 'tool_calls' in " +
    "message %} {%- if not message.tool_calls|length == 1 %} {{- raise_exception(\"This model only " +
    "supports single tool-calls at once!\") }} {%- endif %} {%- set tool_call = message.tool_calls[0]" +
    ".function %} {%- if builtin_tools is defined and tool_call.name in builtin_tools %} {{- " +
    "'<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- \"<|python_tag|>\" + tool_call.name + " +
    "\".call(\" }} {%- for arg_name, arg_val in tool_call.arguments | items %} {{- arg_name + '=\"' + " +
    "arg_val + '\"' }} {%- if not loop.last %} {{- \", \" }} {%- endif %} {%- endfor %} {{- \")\" }} {%- " +
    "else %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- '{\"name\": \"' + " +
    "tool_call.name + '\", ' }} {{- '\"parameters\": ' }} {{- tool_call.arguments | tojson }} {{- \"}\" " +
    "}} {%- endif %} {%- if builtin_tools is defined %} {#- This means we're in ipython mode #} {{- " +
    "\"<|eom_id|>\" }} {%- else %} {{- \"<|eot_id|>\" }} {%- endif %} {%- elif message.role == \"tool\" " +
    "or message.role == \"ipython\" %} {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }} {%- " +
    "if message.content is mapping or message.content is iterable %} {{- message.content | tojson }} {%- " +
    "else %} {{- message.content }} {%- endif %} {{- \"<|eot_id|>\" }} {%- endif %} {%- endfor %} {%- if " +
    "add_generation_prompt %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }} {%- endif %} "
{% endraw %}

val promptAssembler = new PromptAssembler()
  .setInputCol("messages")
  .setOutputCol("prompt")
  .setChatTemplate(template)

promptAssembler.transform(dataDF).select("prompt.result").show(truncate = false)
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello there, how can I help you?<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI need help with organizing my room.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[PromptAssembler](/api/com/johnsnowlabs/nlp/PromptAssembler)
{%- endcapture -%}

{%- capture python_api_link -%}
[PromptAssembler](/api/python/reference/autosummary/sparknlp/base/prompt_assembler/index.html#sparknlp.base.prompt_assembler.PromptAssembler)
{%- endcapture -%}

{%- capture source_link -%}
[PromptAssembler](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/PromptAssembler.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}