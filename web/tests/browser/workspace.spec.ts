import { test,expect } from '@playwright/test';
import { example } from '../../src/example';
import { mkdir } from 'node:fs/promises';
const id='a'.repeat(64);

test('example is a real board, keyboard transport is stable and no analysis is billed',async({page})=>{
 let calls=0;await page.route('**/api/**',route=>{calls++;return route.fulfill({status:401,json:{error:'Preview'}})});
 await page.goto('/');await page.getByRole('button',{name:'Explore example'}).click();
 await expect(page.getByLabel('Replay board')).toBeVisible();
 const next=page.getByRole('button',{name:'Next position',exact:true}),box=await next.boundingBox();
 await next.click();await expect(page.locator('.position-count')).toHaveText('2 / 12');
 await page.keyboard.press('ArrowRight');await expect(page.locator('.position-count')).toHaveText('3 / 12');
 expect(await next.boundingBox()).toEqual(box);
 await page.getByRole('button',{name:'Settings',exact:true}).click();await page.getByLabel('OpenRouter key', {exact:true}).fill('fixture-key');
 await page.keyboard.press('ArrowRight');await expect(page.locator('.position-count')).toHaveText('3 / 12');
 await page.keyboard.press('Escape');await expect(page.getByRole('button',{name:'Settings',exact:true})).toBeFocused();
 expect(calls).toBe(1);
 await mkdir('screenshots',{recursive:true});await page.screenshot({path:'screenshots/replay.png',fullPage:true});
});

test('late analysis cannot replace a different replay position or move the panel',async({page})=>{
 let release:(()=>void)|undefined;
 await page.route('**/api/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/api/session')return route.fulfill({json:{hasKey:true}});
  if(path==='/api/replays')return route.fulfill({json:[{id,title:'Fixture replay',source:'arena',frames:12,created_at:'2026-09-19T00:00:00Z'}]});
  if(path.endsWith('/analysis')){await new Promise<void>(r=>{release=r});return route.fulfill({json:{status:'done',choice:'bolt',probabilities:{pass:.2,bolt:.7,land:.1},cost:.00001,latency:500,input:100,output:10}})}
  return route.fulfill({json:{...example,source:'arena',title:'Fixture replay'}});
 });
 await page.goto('/');await page.getByRole('button',{name:/Fixture replay/}).click();
 const panel=page.getByLabel('Analysis panel'),initial=await panel.boundingBox();
 await page.getByRole('button',{name:'Analyze',exact:true}).click();await expect(page.getByRole('button',{name:'Analyzing…',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Next position',exact:true}).click();await expect(page.locator('.position-count')).toHaveText('2 / 12');
 await expect.poll(()=>!!release).toBeTruthy();release!();
 await page.waitForTimeout(200);expect(await panel.boundingBox()).toEqual(initial);
 await expect(page.locator('.analysis-state')).toContainText('captured choices');
 await expect(page.getByRole('button',{name:'Next position',exact:true})).toBeFocused();
});

test('saved key stays saved on reload and is never placed in browser storage',async({page})=>{
 let saved=false;
 await page.route('**/api/**',route=>{
 const path=new URL(route.request().url()).pathname;
 if(path==='/api/session')return route.fulfill({json:{hasKey:saved}});
 if(path==='/api/key'){saved=true;return route.fulfill({json:{hasKey:true}})}
 return route.fulfill({json:[]});
 });
 await page.goto('/');await page.getByRole('button',{name:'Settings',exact:true}).click();await page.getByLabel('OpenRouter key',{exact:true}).fill('sk-or-test-secret');await page.getByRole('button',{name:'Save key'}).click();
 await page.reload();await page.getByRole('button',{name:'Settings',exact:true}).click();await expect(page.getByLabel('Replace saved key')).toHaveValue('');
 expect(await page.evaluate(()=>JSON.stringify({...localStorage,...sessionStorage}))).not.toContain('test-secret');
});

test('browser Wasm parser matches native Python on the same Arena fixture',async({page})=>{
 await page.goto('/');const expected=await (await page.request.get('/fixtures/arena.expected.json')).json();
 const actual=await page.evaluate(async()=>{
  const text=await (await fetch('/fixtures/arena.log')).text();
  return new Promise((resolve,reject)=>{const worker=new Worker('/parser-worker.js');worker.onerror=reject;worker.onmessage=({data})=>{worker.terminate();data.error?reject(new Error(data.error)):resolve(data.replays)};worker.postMessage({id:1,text,kind:'arena',catalog:{}})});
 });
 expect(actual).toEqual(expected);
});
