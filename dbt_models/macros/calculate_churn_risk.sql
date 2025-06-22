{% macro calculate_churn_risk(score) %}
  CASE 
    WHEN {{ score }} >= 0.8 THEN 'High'
    WHEN {{ score }} >= 0.5 THEN 'Medium'
    ELSE 'Low'
  END
{% endmacro %}
