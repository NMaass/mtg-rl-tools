import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import { example } from './example';
import { nextPosition, safeView, type Replay, type ReplayRow, type Card } from './domain';
import { useAnalysis } from './useAnalysis';
import { Modal } from './Modal';
import { Viewer } from './Viewer';
import { ImportDialog, Settings, errorMessage } from './components';

type OpenReplay = { id: string | null; replay: Replay };
type Dialog = 'import' | 'settings' | null;

export function App() {
  const [rows, setRows] = useState<ReplayRow[]>([]);
  const [opened, setOpened] = useState<OpenReplay | null>(null);
  const [position, setPosition] = useState(0);
  const [seat, setSeat] = useState('1');
  const [dialog, setDialog] = useState<Dialog>(null);
  const [inspect, setInspect] = useState<Card | null>(null);
  const [hasKey, setHasKey] = useState(false);
  const [signedIn, setSignedIn] = useState(false);
  const [notice, setNotice] = useState('');
  const [working, setWorking] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [auto, setAuto] = useState(false);
  const [search, setSearch] = useState('');
  const [art, setArt] = useState<Record<string, string>>({});
  const [imports, setImports] = useState<Replay[]>([]);
  const [ratings, setRatings] = useState<Record<string, string>>({});
  const { results, request, cancel, schedule } = useAnalysis();
  const generation = useRef(0);
  const refresh = useCallback(async () => {
    try {
      const session = await api<{ hasKey: boolean }>('/session');
      setSignedIn(true); setHasKey(session.hasKey);
      setRows(await api<ReplayRow[]>('/replays')); setNotice('');
    } catch { setSignedIn(false); setNotice('Local preview. Hosted storage requires Cloudflare Access.'); }
  }, []);
  useEffect(() => {
    void refresh();
    void fetch('/example-art.json').then(response => response.ok ? response.json() : {}).then(setArt).catch(() => {});
  }, [refresh]);
  const replay = opened?.replay;
  const frame = replay?.frames[position];
  const available = frame ? Object.keys(frame.views) : [];
  const currentSeat = available.includes(seat) ? seat : available[0];
  const view = frame && currentSeat ? safeView(frame.views[currentSeat]) : null;
  const reviewKey = opened?.id + ':' + position + ':' + currentSeat;
  const result = results[reviewKey];

  function pause() { cancel(); setPlaying(false); }
  function showDialog(value: Dialog) { pause(); setDialog(value); }
  function selectReplay(value: Replay, id: string | null) {
    cancel(); generation.current++;
    setOpened({ replay: value, id }); setPosition(0);
    setSeat(Object.keys(value.frames[0].views)[0]); setPlaying(false); setInspect(null); setNotice('');
  }
  async function openSaved(id: string) {
    const version = ++generation.current; setWorking(true);
    try {
      const value = await api<Replay>('/replays/' + id);
      if (version !== generation.current) return;
      selectReplay(value, id);
      void api<{ ratings: { position: number; perspective: string; rating: string }[] }>('/replays/' + id + '/report').then(report => {
        if (report.ratings) setRatings(previous => ({ ...previous, ...Object.fromEntries(report.ratings.map(rating => [id + ':' + rating.position + ':' + rating.perspective, rating.rating])) }));
      }).catch(() => {});
    } catch (error) { setNotice(errorMessage(error)); }
    finally { setWorking(false); }
  }
  function leave() { pause(); setOpened(null); setInspect(null); generation.current++; }
  const move = useCallback((target: number) => {
    if (!opened) return;
    cancel(); setPosition(target); setPlaying(false);
    if (auto && hasKey && opened.id && opened.replay.source !== 'example') {
      const next = opened.replay.frames[target];
      const perspective = Object.keys(next.views).includes(seat) ? seat : Object.keys(next.views)[0];
      if (next.decisions[perspective]?.supported && next.views[perspective]?.priority === perspective) schedule(opened.id, target, perspective);
    }
  }, [opened, auto, hasKey, seat, cancel, schedule]);
  useEffect(() => {
    if (!playing || !replay) return;
    const timer = setInterval(() => setPosition(previous => {
      if (previous >= replay.frames.length - 1) { setPlaying(false); return previous; }
      return previous + 1;
    }), 900);
    return () => clearInterval(timer);
  }, [playing, replay]);
  useEffect(() => {
    function keyboard(event: KeyboardEvent) {
      if (!replay || dialog || inspect || imports.length || event.altKey || event.ctrlKey || event.metaKey || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault(); move(nextPosition(replay, position, event.key === 'ArrowRight' ? 1 : -1, event.shiftKey));
      } else if (event.key === ' ') { event.preventDefault(); cancel(); setPlaying(value => !value); }
      else if (event.key === 'Home' || event.key === 'End') { event.preventDefault(); move(event.key === 'Home' ? 0 : replay.frames.length - 1); }
    }
    window.addEventListener('keydown', keyboard);
    return () => window.removeEventListener('keydown', keyboard);
  }, [replay, dialog, inspect, imports.length, position, move, cancel]);
  async function saveImports() {
    setWorking(true); setNotice('');
    try {
      let last: { id: string } | undefined;
      for (const item of imports) last = await api<{ id: string }>('/replays', 'POST', item);
      const saved = imports[imports.length - 1]; setImports([]); await refresh();
      if (last) selectReplay(saved, last.id);
    } catch (error) { setNotice(errorMessage(error)); }
    finally { setWorking(false); }
  }
  async function rate(value: string) {
    if (!opened?.id) return;
    setWorking(true);
    try {
      await api('/replays/' + opened.id + '/rating', 'PUT', { position, perspective: currentSeat, rating: value });
      setRatings(previous => ({ ...previous, [reviewKey]: value })); setNotice('');
    } catch (error) { setNotice(errorMessage(error)); }
    finally { setWorking(false); }
  }
  async function remove(id: string) {
    setWorking(true);
    try { await api('/replays/' + id, 'DELETE', {}); setRows(previous => previous.filter(row => row.id !== id)); setNotice(''); }
    catch (error) { setNotice(errorMessage(error)); }
    finally { setWorking(false); }
  }

  return <div className="app-shell">
    <header className="app-header"><div className="brand"><span className="brand-mark">P</span><span>PRIORITY</span></div>
      {opened ? <><button className="icon back" aria-label="Back to library" onClick={leave}>←</button><div className="heading"><strong>{replay?.title}</strong><span>{replay?.source === 'example' ? 'Interface example' : replay?.source === 'mtgo' ? 'Public log reconstruction' : 'Recorded replay'}</span></div></> : <div className="heading"><strong>Your replay library</strong><span>Review the position. Understand the choice.</span></div>}
      <div className="header-actions"><button onClick={() => showDialog('import')}>Import</button><button className="icon" aria-label="Settings" onClick={() => showDialog('settings')}>⚙</button></div>
    </header>
    <div className="notice" role="status">{notice}</div>
    {!opened ? <main className="library">
      <div className="library-heading"><div><p className="eyebrow">THE REVIEW ROOM</p><h1>A better look at your game.</h1><p className="muted">Your games, their decisions, and what comes next.</p></div><button className="primary" onClick={() => showDialog('import')}>Import a game <span>↗</span></button></div>
      <div className="library-tools"><h2>Recent replays <span>{rows.length}</span></h2><input type="search" aria-label="Search replays" placeholder="Find a game" value={search} onChange={event => setSearch(event.target.value)} /></div>
      <div className="replay-list">{rows.filter(row => row.title.toLowerCase().includes(search.toLowerCase())).map(row => <article className="replay-row" key={row.id}><button className="replay-open" aria-label={'Open ' + row.title} disabled={working} onClick={() => void openSaved(row.id)}><span className="replay-symbol">↻</span><span><strong>{row.title}</strong><small>{row.source.toUpperCase()} · {row.frames} positions</small></span><time>{new Date(row.created_at).toLocaleDateString()}</time><span>→</span></button><button className="icon delete" aria-label={'Delete ' + row.title} disabled={working} onClick={() => { if (confirm('Delete this saved replay?')) void remove(row.id); }}>×</button></article>)}
        {rows.length === 0 && <div className="empty-library"><span className="empty-mark">◫</span><h3>Your next insight starts here.</h3><p>Import a log, or explore the viewer with an example.</p><button onClick={() => selectReplay(example, null)}>Explore example →</button></div>}
      </div><p className="library-foot">{signedIn ? 'Private to your account · Latest 200 replays' : 'No account needed to preview a local replay.'}</p>
    </main> : frame && view && replay ? <Viewer replay={replay} frame={frame} view={view} decision={frame.decisions[currentSeat]} result={result} position={position} seat={currentSeat} playing={playing} auto={auto} hasKey={hasKey} saved={!!opened.id} busy={working} rating={ratings[reviewKey] || 'unrated'} art={art}
      move={move} changeSeat={value => { pause(); setSeat(value); setInspect(null); }} togglePlay={() => { cancel(); setPlaying(value => !value); }} setAuto={value => { cancel(); setAuto(value); }} settings={() => showDialog('settings')}
      analyze={() => { if (opened.id) request(opened.id, position, currentSeat, result?.status === 'error'); }} inspect={card => { pause(); setInspect(card); }} rate={value => void rate(value)} /> : <main className="fatal"><h2>This perspective is unavailable.</h2><button onClick={leave}>Back to library</button></main>}
    {dialog === 'settings' && <Settings hasKey={hasKey} onSaved={setHasKey} onClose={() => setDialog(null)} />}
    {dialog === 'import' && <ImportDialog onClose={() => setDialog(null)} onImported={items => { setDialog(null); setImports(items); }} />}
    {imports.length > 0 && <Modal title="Ready to review" onClose={() => setImports([])}><p>{imports.length} game{imports.length === 1 ? '' : 's'} parsed on your device.</p><div className="import-summary">{imports.map((item, index) => <div key={index}><strong>{item.title}</strong><span>{item.frames.length} positions</span>{item.warnings.map((warning, j) => <p key={j}>{warning}</p>)}</div>)}</div><p className="hint">Save uploads only the parsed replay to your private library. The original log stays on your device.</p><p role="alert">{notice}</p><footer className="dialog-actions"><button onClick={() => { selectReplay(imports[0], null); setImports([]); }}>Preview locally</button><button className="primary" disabled={!signedIn || working} onClick={() => void saveImports()}>{working ? 'Saving…' : 'Save to library'}</button></footer></Modal>}
    {inspect && <Modal title={inspect.name} onClose={() => setInspect(null)}><p className="card-inspect-type">{inspect.type} <span>{inspect.mana}</span></p><p className="rules-text">{inspect.rules || 'Rules text is not available in this recording.'}</p><p className="hint">{inspect.zone} · {inspect.tapped ? 'Tapped' : 'Untapped'}{inspect.damage ? ' · ' + inspect.damage + ' damage' : ''}{inspect.attachedTo ? ' · Attached to ' + inspect.attachedTo : ''}</p>{Object.entries(inspect.counters).map(([counter, value]) => <p key={counter}>{counter}: {value}</p>)}</Modal>}
  </div>;
}
