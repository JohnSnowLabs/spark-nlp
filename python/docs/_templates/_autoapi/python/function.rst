{% if obj.display %}
.. py:function:: {{ obj.short_name }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}

{% for (args, return_annotation) in obj.overloads %}
              {{ obj.short_name }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}

{% endfor %}
   {% if sphinx_version >= (2, 1) %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}
   {% endif %}

   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}
   {% else %}
   {% endif %}
{% endif %}
