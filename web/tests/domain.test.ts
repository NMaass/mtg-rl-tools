import test from 'node:test';
import assert from 'node:assert/strict';
import { example } from '../src/example';
import { analysisInput,nextPosition,safeView,ReplaySchema,validateAnswer } from '../src/domain';
import { checkOrigin,identity,seal,unseal } from '../worker/security';

test('analysis excludes recorded action, future frames and opponent hand',()=>{
 const f=structuredClone(example.frames[0]);
 f.views['1'].players[1].hand=[{...f.views['1'].players[0].hand[0],name:'SECRET'}];
 const a=analysisInput(f,'1');f.label='FUTURE WIN';f.decisions['1'].chosen=['pass'];
 assert.deepEqual(a,analysisInput(f,'1'));
 assert(!JSON.stringify(a).includes('SECRET'));assert(!JSON.stringify(a).includes('chosen'));
});
test('private knowledge is taken from the selected perspective only',()=>{
 const f=structuredClone(example.frames[0]);
 const other=structuredClone(f.views['1']);other.viewer='2';other.priority='2';other.known=['PRIVATE SCRY'];
 f.views['2']=other;f.decisions['2']=structuredClone(f.decisions['1']);
 assert(!JSON.stringify(analysisInput(f,'1')).includes('PRIVATE SCRY'));
 assert(JSON.stringify(analysisInput(f,'2')).includes('PRIVATE SCRY'));
});
test('unknown priority and multi-selection are not sent',()=>{
 const f=structuredClone(example.frames[0]);f.views['1'].priority=null;assert.throws(()=>analysisInput(f,'1'));
 f.views['1'].priority='1';f.decisions['1'].max=2;assert.throws(()=>analysisInput(f,'1'));
});
test('duplicate actions and bad selections are rejected',()=>{
 const r=structuredClone(example);r.frames[0].decisions['1'].options[1].id='pass';assert(!ReplaySchema.safeParse(r).success);
});
test('face-down card rules and name are not exposed',()=>{
 const view=structuredClone(example.frames[0].views['1']);view.cards[0].faceDown=true;view.cards[0].name='SECRET';view.cards[0].rules='PRIVATE RULES';
 const projected=safeView(view);assert(!JSON.stringify(projected).includes('SECRET'));assert(!JSON.stringify(projected).includes('PRIVATE RULES'));
});
test('choice output must exactly match offered actions',()=>{
 const body={answers:{action:{type:'choice',choice:'a',probabilities:{a:.6,b:.4}}}};
 assert.equal(validateAnswer(body,['a','b']).answers.action.choice,'a');
 assert.throws(()=>validateAnswer(body,['a','c']));
 assert.throws(()=>validateAnswer({answers:{action:{type:'choice',choice:'a',probabilities:{a:NaN,b:0}}}},['a','b']));
});
test('navigation respects boundaries and captured decision stops',()=>{
 assert.equal(nextPosition(example,0,-1),0);assert.equal(nextPosition(example,11,1),11);
 const r=structuredClone(example);r.frames[1].decisions={};assert.equal(nextPosition(r,0,1,true),2);
});
test('encrypted keys are bound to their owner and use unique nonces',async()=>{
 const master=btoa('1'.repeat(32));const encrypted=await seal('sk-or-fixture','alice',master);
 assert(!encrypted.includes('sk-or-fixture'));assert.equal(await unseal(encrypted,'alice',master),'sk-or-fixture');
 assert.notEqual(encrypted,await seal('sk-or-fixture','alice',master));await assert.rejects(()=>unseal(encrypted,'bob',master));
});
test('cross-origin writes are rejected',()=>{
 assert.throws(()=>checkOrigin(new Request('https://app.test/api/key',{method:'PUT',headers:{Origin:'https://evil.test','Content-Type':'application/json'},body:'{}'})));
});
test('local auth bypass cannot be enabled on a hosted URL',async()=>{
 await assert.rejects(()=>identity(new Request('https://priority.workers.dev/api/session'),{DEV_AUTH:'local-test',ACCESS_AUD:'x',ACCESS_ISSUER:'https://REPLACE.cloudflareaccess.com',KEY_ENCRYPTION_KEY:''}));
});
