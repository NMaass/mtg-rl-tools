import { createRemoteJWKSet, jwtVerify } from 'jose';
export interface IdentityEnv { ACCESS_ISSUER:string; ACCESS_AUD:string; DEV_AUTH?:string; KEY_ENCRYPTION_KEY:string }
const jwks=new Map<string,ReturnType<typeof createRemoteJWKSet>>();
export async function digest(value:string) {return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),x=>x.toString(16).padStart(2,'0')).join('')}
export async function identity(request:Request,env:IdentityEnv):Promise<string> {
  const url=new URL(request.url);
  if(env.DEV_AUTH==='local-test' && ['localhost','127.0.0.1'].includes(url.hostname))return 'local-test';
  if(!env.ACCESS_ISSUER || !/^https:\/\/[a-z0-9-]+\.cloudflareaccess\.com$/.test(env.ACCESS_ISSUER)||!env.ACCESS_AUD||env.ACCESS_AUD.startsWith('REPLACE'))throw new Error('Authentication is not configured.');
  const token=request.headers.get('Cf-Access-Jwt-Assertion');
  if(!token)throw new Error('Sign in to access your replays.');
  let keys=jwks.get(env.ACCESS_ISSUER);
  if(!keys){keys=createRemoteJWKSet(new URL(env.ACCESS_ISSUER+'/cdn-cgi/access/certs'));jwks.set(env.ACCESS_ISSUER,keys)}
  const {payload}=await jwtVerify(token,keys,{issuer:env.ACCESS_ISSUER,audience:env.ACCESS_AUD,algorithms:['RS256']});
  if(typeof payload.sub!=='string'||!payload.sub)throw new Error('No user identity.');
  return digest(env.ACCESS_ISSUER+'\0'+payload.sub);
}
export function checkOrigin(request:Request) {
  if(['GET','HEAD'].includes(request.method))return;
  if(request.headers.get('origin')!==new URL(request.url).origin)throw new Error('Cross-origin request refused.');
  if(!request.headers.get('content-type')?.startsWith('application/json'))throw new Error('Expected JSON.');
}
function bytes(text:string) {return Uint8Array.from(atob(text),c=>c.charCodeAt(0))}
async function encryptionKey(secret:string) {
  if(!secret)throw new Error('Key storage is not configured.');
  const raw=bytes(secret);if(raw.length!==32)throw new Error('Invalid server encryption configuration.');
  return crypto.subtle.importKey('raw',raw,'AES-GCM',false,['encrypt','decrypt']);
}
export async function seal(key:string,owner:string,secret:string) {
  const iv=crypto.getRandomValues(new Uint8Array(12));
  const encrypted=await crypto.subtle.encrypt({name:'AES-GCM',iv,additionalData:new TextEncoder().encode(owner)},await encryptionKey(secret),new TextEncoder().encode(key));
  return 'v1.'+btoa(String.fromCharCode(...iv))+'.'+btoa(String.fromCharCode(...new Uint8Array(encrypted)));
}
export async function unseal(value:string,owner:string,secret:string) {
  const [version,iv,data]=value.split('.');if(version!=='v1'||!iv||!data)throw new Error('Unsupported stored key.');
  return new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:bytes(iv),additionalData:new TextEncoder().encode(owner)},await encryptionKey(secret),bytes(data)));
}
export async function readJson(request:Request,max=8*1024*1024):Promise<unknown> {
  const reader=request.body?.getReader();if(!reader)throw new Error('Missing request body.');
  const chunks:Uint8Array[]=[];let length=0;
  try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>max)throw new Error('Request is too large.');chunks.push(value)}}finally{await reader.cancel().catch(()=>{});reader.releaseLock()}
  const buffer=new Uint8Array(length);let offset=0;for(const chunk of chunks){buffer.set(chunk,offset);offset+=chunk.length}
  return JSON.parse(new TextDecoder().decode(buffer));
}
