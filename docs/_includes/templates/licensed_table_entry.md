
{%- capture linkref -%}

{{include.path}}#{{include.name | downcase | replace: ' ', '-' | replace: '(', '' | replace: ')', '' | replace: '_', ''}}

{%- endcapture -%}

|<a href="{{linkref}}">{{include.name}}</a>|{{include.summary}}|