import { useEffect, useRef, useState } from 'react';
import { api } from './api';
import { parseFile } from './importer';
import { Modal } from './Modal';
import { StableText } from './StableSwap';
import type { Card, Replay, View } from './domain';

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.';

export function CardTile({ card, art, onInspect }: { card: Card; art: Record<string, string>; onInspect: (card: Card) => void }) {
  const image = art[card.name];
  return <button className={'game-card ' + (card.tapped ? 'tapped ' : '') + (card.faceDown ? 'hidden-card' : '')} onClick={() => onInspect(card)} aria-label={card.name + (card.tapped ? ', tapped' : '')} title={card.name}>
    <span className="card-art" style={image && !card.faceDown ? { backgroundImage: `url(/example-art/${image}.jpg)` } : undefined}><span className="card-mana">{card.mana.replace(/[{}]/g, '')}</span></span>
    <span className="card-name">{card.name}</span><span className="card-type">{card.type.replace('CardType_', '')}</span>
    <span className="card-bottom"><span>{card.tapped ? 'TAPPED' : card.attacking ? 'ATTACKING' : ''}</span>{card.power !== null && card.toughness !== null && <b>{card.power}/{card.toughness}</b>}</span>
  </button>;
}

export function PlayerBar({ view, seat }: { view: View; seat: string }) {
  const player = view.players.find(p => p.id === seat);
  return <div className="player-bar"><span className={'avatar ' + (seat === view.viewer ? 'hero-avatar' : '')}>{seat === view.viewer ? 'Y' : 'O'}</span><strong>{seat === view.viewer ? 'You' : seat === 'public' ? 'Public view' : 'Opponent'}</strong><span className="muted">{player?.handCount ?? '—'} in hand · {player?.libraryCount ?? '—'} in library</span><span className="priority-chip">{view.priority === seat ? 'Priority' : view.active === seat ? 'Active turn' : ''}</span><b className="life">{player?.life ?? '—'}<small>life</small></b></div>;
}

export function Settings({ hasKey, onSaved, onClose }: { hasKey: boolean; onSaved: (value: boolean) => void; onClose: () => void }) {
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function save(remove = false) {
    setBusy(true); setError('');
    try {
      await api('/key', remove ? 'DELETE' : 'PUT', remove ? {} : { key: key.trim() });
      onSaved(!remove);
      if (mounted.current) { setKey(''); onClose(); }
    } catch (error) { if (mounted.current) setError(errorMessage(error)); }
    finally { if (mounted.current) setBusy(false); }
  }
  return <Modal title="Analysis settings" onClose={onClose}>
    <p>Your OpenRouter key is encrypted on the server. It stays saved between visits.</p>
    <form onSubmit={event => { event.preventDefault(); void save(); }}>
      <label htmlFor="key">{hasKey ? 'Replace saved key' : 'OpenRouter key'}</label>
      <input id="key" type="password" autoComplete="off" placeholder={hasKey ? 'A key is already saved' : 'sk-or-…'} value={key} onChange={event => setKey(event.target.value)} />
      <p className="hint">Only positions you choose to analyze are sent to OpenRouter / TypeSafe. Jev’s scores are not verified win rates.</p>
      <div role="alert" className="form-error">{error}</div>
      <footer className="dialog-actions">{hasKey && <button type="button" className="quiet" disabled={busy} onClick={() => void save(true)}>Remove key</button>}<button className="primary stable-button" disabled={busy || !key.trim()}><StableText value={busy ? 'Saving…' : 'Save key'} candidates={['Saving…', 'Save key']} /></button></footer>
    </form>
  </Modal>;
}

export function ImportDialog({ onClose, onImported }: { onClose: () => void; onImported: (items: Replay[]) => void }) {
  const [kind, setKind] = useState<'arena' | 'mtgo' | 'json'>('arena');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const abort = useRef(new AbortController());
  useEffect(() => () => abort.current.abort(), []);
  async function choose(file?: File) {
    if (!file) return;
    setBusy(true); setError('');
    try {
      const replays = await parseFile(file, kind, abort.current.signal);
      if (!abort.current.signal.aborted) onImported(replays);
    } catch (error) { if (!abort.current.signal.aborted) setError(errorMessage(error)); }
    finally { if (!abort.current.signal.aborted) setBusy(false); }
  }
  return <Modal title="Import a replay" onClose={onClose}>
    <label htmlFor="format">Source</label><select id="format" value={kind} onChange={event => setKind(event.target.value as typeof kind)} disabled={busy}><option value="arena">MTG Arena</option><option value="mtgo">MTGO text game log</option><option value="json">Priority replay JSON</option></select>
    {kind === 'arena' ? <><p>In Arena: <strong>Settings → View Account → Detailed Logs (Plugin Support)</strong>. Restart Arena, then play a game.</p><details><summary>Where is Player.log?</summary><p>Windows</p><code>{'%USERPROFILE%\\AppData\\LocalLow\\Wizards Of The Coast\\MTGA\\Player.log'}</code><p>macOS</p><code>~/Library/Logs/Wizards Of The Coast/MTGA/Player.log</code><p>Logs can be replaced when Arena restarts. Import them before starting another session.</p></details></> : kind === 'mtgo' ? <p>Choose a copied or exported <strong>text game log</strong>. Native binary .dat files are not supported. Public logs do not contain complete hands or priority choices.</p> : <p>Choose an exported Priority replay. Only perspectives present in the recording will be available.</p>}
    <p className="hint">You choose the file. Parsing stays on this device; nothing uploads until you select Save.</p>
    <div role="alert" className="form-error">{error}</div>
    <label className={'file-button primary ' + (busy ? 'disabled' : '')}>{busy ? 'Reading log…' : 'Choose file'}<input type="file" accept={kind === 'json' ? '.json' : '.log,.txt'} disabled={busy} onChange={event => void choose(event.target.files?.[0])} /></label>
  </Modal>;
}
