CREATE TABLE IF NOT EXISTS user_keys (owner TEXT PRIMARY KEY, ciphertext TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL, source TEXT NOT NULL, frames INTEGER NOT NULL, created_at TEXT NOT NULL, object_key TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS replays_owner_date ON replays(owner, created_at DESC);
CREATE TABLE IF NOT EXISTS analyses (owner TEXT NOT NULL, cache_key TEXT NOT NULL, replay_id TEXT NOT NULL, status TEXT NOT NULL, result TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(owner, cache_key));
CREATE INDEX IF NOT EXISTS analyses_replay ON analyses(owner, replay_id);
CREATE TABLE IF NOT EXISTS analysis_budget (owner TEXT NOT NULL, day TEXT NOT NULL, calls INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(owner, day));
CREATE TABLE IF NOT EXISTS ratings (owner TEXT NOT NULL, replay_id TEXT NOT NULL, position INTEGER NOT NULL, perspective TEXT NOT NULL, rating TEXT NOT NULL, PRIMARY KEY(owner,replay_id,position,perspective));
