export class ApiError extends Error { constructor(message:string,readonly status:number){super(message)} }
export async function api<T>(path:string,method='GET',body?:unknown):Promise<T> {
  const response=await fetch('/api'+path,{method,credentials:'same-origin',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const value:unknown=await response.json();
  if(!response.ok){const message=typeof value==='object'&&value!==null&&'error' in value?String(value.error):'Request failed.';throw new ApiError(message,response.status)}
  return value as T;
}
