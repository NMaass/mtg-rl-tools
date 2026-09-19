CREATE TABLE IF NOT EXISTS analysis_attempts (
  id TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  replay_id TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  result TEXT
);
CREATE INDEX IF NOT EXISTS attempts_replay ON analysis_attempts(owner,replay_id,started_at);
