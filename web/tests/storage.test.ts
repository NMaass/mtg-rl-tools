import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { getPlatformProxy } from 'wrangler';
import worker, { type Env } from '../worker/index';
import { example } from '../src/example';

test('real local D1/R2 persist keys, isolate owners, retain charged failures and cache success', async () => {
  const platform = await getPlatformProxy<Env>({ configPath: 'wrangler.jsonc', persist: false });
  const env = { ...platform.env, DEV_AUTH: 'local-test', KEY_ENCRYPTION_KEY: btoa('s'.repeat(32)) };
  const originalFetch = globalThis.fetch;
  try {
    for (const file of (await readdir('migrations')).filter(f=>f.endsWith('.sql')).sort()) {
      for (const statement of (await readFile('migrations/' + file, 'utf8')).split(';').filter(s=>s.trim())) await env.DB.prepare(statement).run();
    }
    const call = (path: string, method = 'GET', body?: unknown) => worker.fetch(new Request('http://localhost/api' + path, { method, headers: { Origin: 'http://localhost', 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) }), env);
    assert.equal((await call('/key', 'PUT', { key: 'sk-or-fixture-secret' })).status, 200);
    const stored = await env.DB.prepare('SELECT ciphertext FROM user_keys').first<{ ciphertext: string }>();
    assert(stored); assert(!stored.ciphertext.includes('sk-or'));
    assert.deepEqual(await (await call('/session')).json(), { hasKey: true, engine: false });
    const replay = { ...structuredClone(example), source: 'arena' };
    const created = await (await call('/replays', 'POST', replay)).json() as { id: string };
    assert(created.id);
    assert.deepEqual(await (await call('/replays/' + created.id)).json(), replay);
    const other = 'b'.repeat(64);
    await env.DB.prepare('INSERT INTO replays(id,owner,title,source,frames,created_at,object_key) VALUES(?,?,?,?,?,?,?)').bind(other, 'other-user', 'Private', 'arena', 1, '', 'private.json').run();
    assert.equal((await call('/replays/' + other)).status, 400);
    assert.equal((await call('/replays/' + other, 'DELETE', {})).status, 400);
    assert.equal((await call('/replays/' + other + '/analysis', 'POST', { position: 0, perspective: '1' })).status, 400);

    let calls = 0;
    globalThis.fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (url === 'https://openrouter.ai/api/alpha/decisions') {
        calls++;
        assert.equal(new Headers(init?.headers).get('Authorization'), 'Bearer sk-or-fixture-secret');
        return Response.json({ answers: { action: { type: 'choice', choice: calls===1?'invalid':'pass', probabilities: { pass: .6, bolt: .3, land: .1 } } }, usage: { cost: .00001, input_tokens: 100, output_tokens: 5 }, model: 'typesafe/jev-1.13' });
      }
      return originalFetch(input, init);
    };
    const args = { position: 0, perspective: '1' };
    const failed = await (await call('/replays/' + created.id + '/analysis', 'POST', args)).json() as { status: string; cost: number };
    assert.equal(failed.status, 'error'); assert.equal(failed.cost, .00001);
    await call('/replays/' + created.id + '/analysis', 'POST', args); assert.equal(calls, 1);
    const answer = await (await call('/replays/' + created.id + '/analysis', 'POST', { ...args, retry: true })).json() as { status: string; choice: string };
    assert.equal(answer.status, 'done'); assert.equal(answer.choice, 'pass');
    await call('/replays/' + created.id + '/analysis', 'POST', args); assert.equal(calls, 2);
    await call('/replays/' + created.id + '/rating', 'PUT', { ...args, rating: 'useful' });
    const report = await (await call('/replays/' + created.id + '/report')).json() as { attempts: { result: string }[]; ratings: { rating: string }[] };
    assert.equal(report.attempts.length, 2);
    assert.equal(report.attempts.reduce((sum, a)=>sum+JSON.parse(a.result).cost,0), .00002);
    assert.equal(report.ratings[0].rating, 'useful');
    assert(!JSON.stringify(report).includes('sk-or-fixture-secret'));
    await call('/replays/' + created.id, 'DELETE', {});
    assert.equal((await call('/replays/' + created.id)).status, 400);
    assert.equal((await env.DB.prepare('SELECT COUNT(*) AS count FROM analysis_attempts').first<{count:number}>())?.count, 0);
    await call('/key', 'DELETE', {});
    assert.deepEqual(await (await call('/session')).json(), { hasKey: false, engine: false });
  } finally { globalThis.fetch = originalFetch; await platform.dispose(); }
});
