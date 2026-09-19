import { ReplaySchema, type Replay } from './domain';
export async function parseFile(file:File, kind:'arena'|'mtgo'|'json', signal?:AbortSignal):Promise<Replay[]> {
  if(file.size>32*1024*1024) throw new Error('Choose a log smaller than 32 MB or split it into sessions.');
  const text=await file.text();
  if(kind==='json') {
    const value:unknown=JSON.parse(text);
    return (Array.isArray(value)?value:[value]).map(v=>ReplaySchema.parse(v));
  }
  const catalog=await fetch('/catalog.json').then(async r=>r.ok?r.json():{}).catch(()=>({}));
  return new Promise((resolve,reject)=>{
    const worker=new Worker('/parser-worker.js');
    const cleanup=()=>{worker.terminate();clearTimeout(timer);signal?.removeEventListener('abort',abort)};
    const abort=()=>{cleanup();reject(new DOMException('Import cancelled','AbortError'))};
    const timer=setTimeout(()=>{cleanup();reject(new Error('Import timed out. Try a smaller log.'))},90000);
    if(signal?.aborted){abort();return}
    signal?.addEventListener('abort',abort,{once:true});
    worker.onerror=()=>{cleanup();reject(new Error('The local parser could not start. Reload and retry.'))};
    worker.onmessage=({data})=>{cleanup();try{if(data.error)throw new Error(data.error);resolve((data.replays as unknown[]).map(r=>ReplaySchema.parse(r)))}catch(e){reject(e)}};
    worker.postMessage({id:1,text,kind,catalog});
  });
}
