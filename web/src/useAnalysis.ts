import { useCallback,useEffect,useRef,useState } from 'react';
import { api } from './api';
import type { Analysis } from './domain';
interface Job { id:string;position:number;seat:string;key:string;retry:boolean;epoch:number }
export function useAnalysis(){
  const [results,setResults]=useState<Record<string,Analysis>>({});
  const pending=useRef<Job|null>(null),running=useRef(false),epoch=useRef(0),cache=useRef<Record<string,Analysis>>({}),mounted=useRef(true);
  const scheduled=useRef<ReturnType<typeof setTimeout>|null>(null);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;pending.current=null;epoch.current++;if(scheduled.current)clearTimeout(scheduled.current)}},[]);
  const drain=useCallback(async()=>{
    if(running.current)return;running.current=true;
    try{
      while(pending.current){
        const job:Job=pending.current;pending.current=null;
        if(job.epoch!==epoch.current)continue;
        let result:Analysis;
        try{result=await api<Analysis>('/replays/'+job.id+'/analysis','POST',{position:job.position,perspective:job.seat,retry:job.retry})}
        catch(error){result={status:'error',cost:null,latency:null,input:null,output:null,error:error instanceof Error?error.message:'Analysis failed.'}}
        cache.current[job.key]=result;
        if(mounted.current)setResults(r=>({...r,[job.key]:result}));
      }
    }finally{running.current=false}
  },[]);
  const request=useCallback((id:string,position:number,seat:string,retry=false)=>{
    const key=id+':'+position+':'+seat,previous=cache.current[key];
    if(previous&&!(retry&&previous.status==='error'))return;
    const loading:Analysis={status:'pending',cost:null,latency:null,input:null,output:null};
    cache.current[key]=loading;setResults(r=>({...r,[key]:loading}));
    if(pending.current){const old=pending.current.key;delete cache.current[old];setResults(r=>{const next={...r};delete next[old];return next})}
    pending.current={id,position,seat,key,retry,epoch:epoch.current};void drain();
  },[drain]);
  const cancel=useCallback(()=>{
    epoch.current++;
    if(scheduled.current){clearTimeout(scheduled.current);scheduled.current=null}
    if(pending.current){const key=pending.current.key;delete cache.current[key];setResults(r=>{const next={...r};delete next[key];return next})}
    pending.current=null;
  },[]);
  const schedule=useCallback((id:string,position:number,seat:string)=>{
    if(scheduled.current)clearTimeout(scheduled.current);
    const version=epoch.current;
    scheduled.current=setTimeout(()=>{scheduled.current=null;if(version===epoch.current)request(id,position,seat)},100);
  },[request]);
  return {results,request,cancel,schedule};
}
