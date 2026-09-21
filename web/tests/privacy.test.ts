import test from 'node:test';
import assert from 'node:assert/strict';
import { safeView, analysisInput } from '../src/domain';
import { example } from '../src/example';

test('face-down reference names and attachment references cannot leak identity',()=>{
  const view=structuredClone(example.frames[0].views['1']);
  view.cards[0].faceDown=true;view.cards[0].name='SECRET';view.cards[0].ref='SECRET copy 1';
  view.cards[1].attachedTo='SECRET copy 1';
  const result=safeView(view);
  assert(!JSON.stringify(result).includes('SECRET'));
  assert.equal(result.cards[1].attachedTo,result.cards[0].ref);
});
test('a private hand card cannot enter the public-zone array',()=>{
  const view=structuredClone(example.frames[0].views['1']);
  view.cards.push({...view.players[0].hand[0],name:'SECRET HAND',owner:'2'});
  assert(!JSON.stringify(safeView(view)).includes('SECRET HAND'));
});
test('unsupported prompt types cannot enable analysis with a forged supported flag',()=>{
  const frame=structuredClone(example.frames[0]);frame.decisions['1'].kind='MULLIGAN';
  assert.throws(()=>analysisInput(frame,'1'));
});
