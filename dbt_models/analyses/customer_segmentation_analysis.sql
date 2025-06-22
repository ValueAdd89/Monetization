SELECT
  CASE
    WHEN lifetime_value > 1000 THEN 'High Value'
    WHEN lifetime_value BETWEEN 500 AND 1000 THEN 'Medium Value'
    ELSE 'Low Value'
  END AS segment,
  COUNT(*) AS customer_count
FROM mart_customer_360
GROUP BY 1
ORDER BY 2 DESC;
