import { z } from 'zod';
import { ReplaySchema, analysisInput, validateAnswer, type Analysis, type Replay } from '../src/domain';
import { identity, checkOrigin, digest, seal, unseal, readJson, type IdentityEnv } from './security';

export interface Env extends IdentityEnv { DB: D1Database; REPLAYS: R2Bucket; ASSETS: Fetcher }
const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });
const empty: Analysis = { status: 'pending', cost: null, latency: null, input: null, output: null };

async function owned(env: Env, owner: string, id: string): Promise<Replay> {
  const row = await env.DB.prepare('SELECT object_key FROM replays WHERE owner=? AND id=?').bind(owner, id).first<{ object_key: string }>();
  if (!row) throw new Error('Replay not found.');
  const object = await env.REPLAYS.get(row.object_key);
  if (!object) throw new Error('Replay data is unavailable.');
  return ReplaySchema.parse(await object.json());
}

function nullableUsage(value: unknown): Pick<Analysis, 'cost' | 'input' | 'output'> {
  const parsed = z.object({ usage: z.object({ cost: z.number().finite().nonnegative().optional(), input_tokens: z.number().int().nonnegative().optional(), output_tokens: z.number().int().nonnegative().optional() }).optional() }).safeParse(value);
  const usage = parsed.success ? parsed.data.usage : undefined;
  return { cost: usage?.cost ?? null, input: usage?.input_tokens ?? null, output: usage?.output_tokens ?? null };
}

async function analyze(request: Request, env: Env, owner: string, id: string): Promise<Response> {
  const args = z.object({ position: z.number().int().nonnegative(), perspective: z.string().max(160), retry: z.boolean().default(false) }).parse(await readJson(request, 2048));
  const replay = await owned(env, owner, id);
  if (replay.source === 'example') return json({ error: 'Examples do not use paid analysis.' }, 400);
  const frame = replay.frames[args.position];
  if (!frame) return json({ error: 'Position not found.' }, 404);
  const input = analysisInput(frame, args.perspective);
  const cache = await digest(JSON.stringify(input));
  const now = new Date().toISOString();
  const old = await env.DB.prepare('SELECT status,result,updated_at FROM analyses WHERE owner=? AND cache_key=?').bind(owner, cache).first<{ status: string; result: string | null; updated_at: string }>();
  const stale = old?.status === 'pending' && Date.now() - Date.parse(old.updated_at) > 120000;
  if (old && !(args.retry && (old.status === 'error' || stale))) {
    if (old.result) return json(JSON.parse(old.result));
    return json({ ...empty, status: stale ? 'error' : 'pending', error: stale ? 'The previous call did not finish recording. Its cost is unknown. An explicit retry may cause another charge.' : 'Analysis is still running. No duplicate call was sent.' });
  }
  const storedKey = await env.DB.prepare('SELECT ciphertext FROM user_keys WHERE owner=?').bind(owner).first<{ ciphertext: string }>();
  if (!storedKey) return json({ error: 'Add your OpenRouter key in Settings.' }, 428);
  const lock = old
    ? await env.DB.prepare("UPDATE analyses SET status='pending',result=NULL,updated_at=? WHERE owner=? AND cache_key=? AND updated_at=?").bind(now, owner, cache, old.updated_at).run()
    : await env.DB.prepare("INSERT OR IGNORE INTO analyses(owner,cache_key,replay_id,status,updated_at) VALUES(?,?,?,'pending',?)").bind(owner, cache, id, now).run();
  if (lock.meta.changes !== 1) return json(empty);

  const attempt = crypto.randomUUID();
  await env.DB.prepare("INSERT INTO analysis_attempts(id,owner,replay_id,cache_key,status,started_at) VALUES(?,?,?,?,'pending',?)").bind(attempt, owner, id, cache, now).run();
  let result: Analysis = { ...empty, status: 'error', error: 'Analysis failed. No automatic retry was sent.' };
  const started = performance.now();
  try {
    const budget = await env.DB.prepare('INSERT INTO analysis_budget(owner,day,calls) VALUES(?,?,1) ON CONFLICT(owner,day) DO UPDATE SET calls=calls+1 WHERE calls<500 RETURNING calls').bind(owner, now.slice(0, 10)).first();
    if (!budget) {
      result = { ...result, cost: 0, error: 'Daily analysis limit reached. No provider call was sent.' };
    } else {
      const apiKey = await unseal(storedKey.ciphertext, owner, env.KEY_ENCRYPTION_KEY);
      const response = await fetch('https://openrouter.ai/api/alpha/decisions', { method: 'POST', redirect: 'error', headers: { Authorization: 'Bearer ' + apiKey, 'Content-Type': 'application/json' }, body: JSON.stringify(input), signal: AbortSignal.timeout(20000) });
      if (!response.ok) {
        result.error = ({ 401: 'OpenRouter rejected the key.', 402: 'OpenRouter credits are exhausted.', 429: 'OpenRouter rate limit reached.' } as Record<number, string>)[response.status] || 'Provider request failed. Cost may be unknown.';
        await response.body?.cancel();
      } else {
        const body = await readJson(response, 1024 * 1024);
        result = { ...result, ...nullableUsage(body) };
        const answer = validateAnswer(body, Object.keys(input.questions.action.criteria));
        result = { ...result, status: 'done', error: undefined, choice: answer.answers.action.choice, probabilities: answer.answers.action.probabilities, model: answer.model?.replaceAll(apiKey, '[redacted]').slice(0, 160) };
      }
    }
  } catch {
    result.error = 'Network error or invalid provider answer. Any reported cost is retained; missing cost is unknown. Retry explicitly.';
  }
  result.latency = Math.round(performance.now() - started);
  await env.DB.batch([
    env.DB.prepare('UPDATE analysis_attempts SET status=?,result=? WHERE owner=? AND id=?').bind(result.status, JSON.stringify(result), owner, attempt),
    env.DB.prepare('UPDATE analyses SET status=?,result=?,updated_at=? WHERE owner=? AND cache_key=? AND updated_at=?').bind(result.status, JSON.stringify(result), new Date().toISOString(), owner, cache, now)
  ]);
  return json(result);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (!url.pathname.startsWith('/api/')) return env.ASSETS.fetch(request);
    let owner: string;
    try { owner = await identity(request, env); }
    catch { return json({ error: 'Sign in is required. Configure Cloudflare Access before using the hosted API.' }, 401); }
    try {
      checkOrigin(request);
      if (url.pathname === '/api/session' && request.method === 'GET') {
        const key = await env.DB.prepare('SELECT updated_at FROM user_keys WHERE owner=?').bind(owner).first();
        return json({ hasKey: !!key, engine: false });
      }
      if (url.pathname === '/api/key') {
        if (request.method === 'PUT') {
          const { key } = z.object({ key: z.string().min(10).max(2048).regex(/^\S+$/) }).parse(await readJson(request, 4096));
          const ciphertext = await seal(key, owner, env.KEY_ENCRYPTION_KEY);
          await env.DB.prepare('INSERT INTO user_keys(owner,ciphertext,updated_at) VALUES(?,?,?) ON CONFLICT(owner) DO UPDATE SET ciphertext=excluded.ciphertext,updated_at=excluded.updated_at').bind(owner, ciphertext, new Date().toISOString()).run();
          return json({ hasKey: true });
        }
        if (request.method === 'DELETE') {
          await env.DB.prepare('DELETE FROM user_keys WHERE owner=?').bind(owner).run();
          return json({ hasKey: false });
        }
      }
      if (url.pathname === '/api/replays') {
        if (request.method === 'GET') return json((await env.DB.prepare('SELECT id,title,source,frames,created_at FROM replays WHERE owner=? ORDER BY created_at DESC LIMIT 200').bind(owner).all()).results);
        if (request.method === 'POST') {
          const replay = ReplaySchema.parse(await readJson(request));
          if (replay.source === 'example') return json({ error: 'Do not upload interface examples.' }, 400);
          const serialized = JSON.stringify(replay);
          const id = await digest(owner + '\0' + serialized);
          const objectKey = owner + '/' + id + '.json';
          await env.REPLAYS.put(objectKey, serialized, { httpMetadata: { contentType: 'application/json' } });
          await env.DB.prepare('INSERT OR IGNORE INTO replays(id,owner,title,source,frames,created_at,object_key) VALUES(?,?,?,?,?,?,?)').bind(id, owner, replay.title, replay.source, replay.frames.length, new Date().toISOString(), objectKey).run();
          return json({ id }, 201);
        }
      }
      const match = url.pathname.match(/^\/api\/replays\/([a-f0-9]{64})(?:\/(analysis|rating|report))?$/);
      if (match) {
        const [, id, action] = match;
        if (action === 'analysis' && request.method === 'POST') return await analyze(request, env, owner, id);
        if (action === 'report' && request.method === 'GET') {
          await owned(env, owner, id);
          const ratings = (await env.DB.prepare('SELECT position,perspective,rating FROM ratings WHERE owner=? AND replay_id=?').bind(owner, id).all()).results;
          const attempts = (await env.DB.prepare('SELECT id,status,result,started_at FROM analysis_attempts WHERE owner=? AND replay_id=? ORDER BY started_at').bind(owner, id).all()).results;
          return json({ ratings, attempts });
        }
        if (action === 'rating' && request.method === 'PUT') {
          const replay = await owned(env, owner, id);
          const args = z.object({ position: z.number().int().nonnegative(), perspective: z.string().max(160), rating: z.enum(['useful', 'wrong', 'unsure', 'unrated']) }).parse(await readJson(request, 1024));
          if (!replay.frames[args.position]?.views[args.perspective]) return json({ error: 'Perspective not found.' }, 404);
          await env.DB.prepare('INSERT INTO ratings(owner,replay_id,position,perspective,rating) VALUES(?,?,?,?,?) ON CONFLICT(owner,replay_id,position,perspective) DO UPDATE SET rating=excluded.rating').bind(owner, id, args.position, args.perspective, args.rating).run();
          return json({ saved: true });
        }
        if (!action && request.method === 'GET') return json(await owned(env, owner, id));
        if (!action && request.method === 'DELETE') {
          await owned(env, owner, id);
          await env.DB.batch([
            env.DB.prepare('DELETE FROM analyses WHERE owner=? AND replay_id=?').bind(owner, id),
            env.DB.prepare('DELETE FROM analysis_attempts WHERE owner=? AND replay_id=?').bind(owner, id),
            env.DB.prepare('DELETE FROM ratings WHERE owner=? AND replay_id=?').bind(owner, id),
            env.DB.prepare('DELETE FROM replays WHERE owner=? AND id=?').bind(owner, id)
          ]);
          await env.REPLAYS.delete(owner + '/' + id + '.json');
          return json({ deleted: true });
        }
      }
      return json({ error: 'Not found.' }, 404);
    } catch (error) {
      const visible = ['Replay not found.', 'Replay data is unavailable.', 'Request is too large.', 'Cross-origin request refused.'];
      return json({ error: error instanceof z.ZodError ? 'Invalid data format.' : error instanceof Error && visible.includes(error.message) ? error.message : 'The request could not be completed. No credentials were logged.' }, 400);
    }
  }
};
