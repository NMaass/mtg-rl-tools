import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getPlatformProxy } from 'wrangler';
import worker,{type Env} from '../worker/index';
import { example } from '../src/example';

test('real local D1 and R2 round-trip, encrypted key, cache and deletion',async()=>{
 const platform=await getPlatformProxy<Env>({configPath:'wrangler.jsonc',persist:false});
 const env={...platform.env,DEV_AUTH:'local-test',KEY_ENCRYPTION_KEY:btoa('s'.repeat(32))};
 const originalFetch=globalThis.fetch;
 try{
  const sql=await readFile('migrations/0001.sql','utf8');
  for(const statement of sql.split(';').filter(s=>s.trim()))await env.DB.prepare(statement).run();
  const call=(path:string,method='GET',body?:unknown)=>worker.fetch(new Request('http://localhost/api'+path,{method,headers:{Origin:'http://localhost','Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)}),env);
  assert.equal((await call('/key','PUT',{key:'sk-or-fixture-secret'})).status,200);
  const stored=await env.DB.prepare('SELECT ciphertext FROM user_keys').first<{ciphertext:string}>();assert(stored);assert(!stored.ciphertext.includes('sk-or'));
  assert.deepEqual(await (await call('/session')).json(),{hasKey:true,engine:false});
  const replay={...structuredClone(example),source:'arena'};
  const created=await (await call('/replays','POST',replay)).json() as {id:string};assert(created.id);
  assert.deepEqual(await (await call('/replays/'+created.id)).json(),replay);
  let calls=0;
  globalThis.fetch=async(input,init)=>{
   const url=typeof input==='string'?input:input instanceof URL?input.href:input.url;
   if(url==='https://openrouter.ai/api/alpha/decisions'){
    calls++;assert.equal(new Headers(init?.headers).get('Authorization'),'Bearer sk-or-fixture-secret');
    return Response.json({answers:{action:{type:'choice',choice:'pass',probabilities:{pass:.6,bolt:.3,land:.1}}},usage:{cost:.00001,input_tokens:100,output_tokens:5},model:'typesafe/jev-1.13'});
   }
   return originalFetch(input,init);
  };
  const request={position:0,perspective:'1'};
  const answer=await (await call('/replays/'+created.id+'/analysis','POST',request)).json() as {status:string;choice:string};
  assert.equal(answer.status,'done');assert.equal(answer.choice,'pass');
  await call('/replays/'+created.id+'/analysis','POST',request);assert.equal(calls,1);
  await call('/replays/'+created.id+'/rating','PUT',{...request,rating:'useful'});
  assert.equal((await env.DB.prepare('SELECT rating FROM ratings').first<{rating:string}>())?.rating,'useful');
  await call('/replays/'+created.id,'DELETE',{});assert.equal((await call('/replays/'+created.id)).status,400);
  await call('/key','DELETE',{});assert.deepEqual(await (await call('/session')).json(),{hasKey:false,engine:false});
 }finally{globalThis.fetch=originalFetch;await platform.dispose()}
});
