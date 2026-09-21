let runtime;
async function load() {
  if (!runtime) runtime=(async()=>{
    importScripts('/parser/pyodide.js');
    const py=await loadPyodide({indexURL:'/parser/'});
    const response=await fetch('/parser/sources.json');
    if(!response.ok) throw new Error('Parser assets are unavailable.');
    const sources=await response.json();
    for(const [path,source] of Object.entries(sources)) {
      py.FS.mkdirTree(path.slice(0,path.lastIndexOf('/')));py.FS.writeFile(path,source);
    }
    py.runPython("import sys\nsys.path.insert(0,'/app')\nfrom magic_cabt.browser_export import parse_export");
    return py;
  })().catch(error=>{runtime=undefined;throw error});
  return runtime;
}
self.onmessage=async({data})=>{
  try {
    const py=await load();
    py.globals.set('source_text',data.text);py.globals.set('source_kind',data.kind);py.globals.set('card_catalog',JSON.stringify(data.catalog||{}));
    const json=py.runPython('parse_export(source_text,source_kind,card_catalog)');
    self.postMessage({id:data.id,replays:JSON.parse(json)});
    py.globals.delete('source_text');py.globals.delete('card_catalog');
  } catch(error) {self.postMessage({id:data.id,error:'Import failed. '+String(error.message||error).slice(0,300)});}
};
