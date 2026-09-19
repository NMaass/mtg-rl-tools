import type { ElementType, ReactNode } from 'react';

export interface StableSwapState<K extends string> {
  key: K;
  content: ReactNode;
}

export interface StableSwapProps<K extends string> {
  active: K;
  states: readonly StableSwapState<K>[];
  axis?: 'both' | 'height';
  as?: ElementType;
  className?: string;
}

export function StableSwap<K extends string>({active,states,axis='both',as,className}:StableSwapProps<K>) {
  if (!states.some((state)=>state.key===active)) throw new Error(`StableSwap received undeclared active state: ${active}`);
  const Root=as??(axis==='height'?'div':'span');
  const Cell=axis==='height'?'div':'span';
  return <Root className={className} data-stable-swap data-axis={axis}>{states.map(state=>{
    const isActive=state.key===active;
    return <Cell key={state.key} data-stable-swap-state data-active={isActive?'':undefined} aria-hidden={isActive?undefined:true}>{state.content}</Cell>;
  })}</Root>;
}
export interface StableTextProps {value:string;candidates:readonly string[];render?:(value:string)=>ReactNode;className?:string}
export function StableText({value,candidates,render=(candidate)=>candidate,className}:StableTextProps){
  const values=Array.from(new Set([...candidates,value]));
  return <StableSwap active={value} className={className} states={values.map(candidate=>({key:candidate,content:render(candidate)}))}/>;
}
