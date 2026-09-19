import { readFile } from 'node:fs/promises';
const config=JSON.parse(await readFile(new URL('../wrangler.jsonc',import.meta.url),'utf8'));
const catalog=JSON.parse(await readFile(new URL('../public/catalog.json',import.meta.url),'utf8'));
if(JSON.stringify(config).includes('REPLACE_'))throw new Error('Set the D1 ID and Cloudflare Access issuer/audience in wrangler.jsonc before deploying.');
if(!Object.keys(catalog.arena||{}).length)throw new Error('Build a real card catalogue before deploying.');
if(config.vars?.DEV_AUTH)throw new Error('Never deploy DEV_AUTH.');
console.log('Bindings and card catalogue configured. Confirm KEY_ENCRYPTION_KEY and Access policy separately.');
