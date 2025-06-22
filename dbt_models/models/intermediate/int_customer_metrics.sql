SELECT
  user_id,
  COUNT(DISTINCT session_id) AS total_sessions,
  AVG(session_length) AS avg_session_time
FROM stg_usage
GROUP BY user_id
