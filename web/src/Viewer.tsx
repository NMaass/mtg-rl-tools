import { useEffect, useRef } from 'react';
import { CardTile } from './components';
import { PlayerBar } from './PlayerBar';
import { humanPhase, nextPosition, type Analysis, type Card, type Decision, type Frame, type Replay, type View } from './domain';
import { StableText } from './StableSwap';
import './viewport.css';

interface ViewerProps {
  replay: Replay; frame: Frame; view: View; decision?: Decision; result?: Analysis;
  position: number; seat: string; playing: boolean; auto: boolean; hasKey: boolean; saved: boolean;
  busy: boolean; rating: string; art: Record<string, string>;
  move: (position: number) => void; changeSeat: (seat: string) => void; togglePlay: () => void;
  setAuto: (value: boolean) => void; analyze: () => void; settings: () => void;
  inspect: (card: Card) => void; rate: (value: string) => void;
}

export function Viewer(props: ViewerProps) {
  const { replay, frame, view, decision, result, position, seat, playing, auto, hasKey, saved, busy, rating, art, move, changeSeat, togglePlay, setAuto, analyze, settings, inspect, rate } = props;
  const history = useRef<HTMLOListElement>(null);
  const sample = replay.source === 'example';
  const eligible = saved && !sample && view.priority === seat && !!decision?.supported && decision.min === 1 && decision.max === 1;
  const bottom = view.viewer === 'public' ? view.players[0].id : view.viewer;
  const opponent = view.players.find(player => player.id !== bottom)?.id;
  const options = [...(decision?.options ?? [])].sort((a, b) => (result?.probabilities?.[b.id] ?? 0) - (result?.probabilities?.[a.id] ?? 0));
  useEffect(() => {
    const list = history.current;
    const node = list?.querySelector<HTMLElement>('[aria-current="step"]');
    if (!list || !node) return;
    const top = node.offsetTop - list.offsetTop;
    if (top < list.scrollTop || top + node.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = Math.max(0, top - list.clientHeight / 2);
  }, [position]);
  const tiles = (cards: Card[]) => cards.map(card => <CardTile key={card.ref} card={card} art={art} onInspect={inspect} />);
  const heroField = view.cards.filter(card => card.zone === 'battlefield' && card.controller === bottom);
  const opponentField = view.cards.filter(card => card.zone === 'battlefield' && card.controller !== bottom);
  const stack = view.cards.filter(card => card.zone === 'stack');
  const hand = view.players.find(player => player.id === view.viewer)?.hand ?? [];
  const other = view.cards.filter(card => !['battlefield', 'stack'].includes(card.zone));
  return <main className="workspace">
    <section className="board-column" aria-label="Replay board">
      <div className="board-toolbar"><span className="eyebrow">TURN {frame.turn}</span><strong>{humanPhase(frame.phase) || 'Position'}</strong><select aria-label="Perspective" value={seat} onChange={event => changeSeat(event.target.value)}>{Object.keys(frame.views).map(p => <option key={p} value={p}>{p === 'public' ? 'Public view' : 'Player ' + p + ' perspective'}</option>)}</select></div>
      <div className="battlefield">
        <div className="board-half opponent-half">{opponent && <PlayerBar view={view} seat={opponent} />}<div className="card-lane">{tiles(opponentField)}{!opponentField.length && <span className="zone-empty">Opponent battlefield</span>}</div></div>
        <div className="stack-lane"><span>STACK</span>{stack.length ? stack.map(card => <button key={card.ref} onClick={() => inspect(card)}>{card.name}</button>) : <small>Empty</small>}</div>
        <div className="board-half hero-half"><div className="card-lane">{tiles(heroField)}{!heroField.length && <span className="zone-empty">{view.viewer === 'public' ? 'Player 1 battlefield' : 'Your battlefield'}</span>}</div><PlayerBar view={view} seat={bottom} /></div>
      </div>
      <div className="hand-area"><span className="eyebrow">{view.viewer === 'public' ? 'PUBLIC RECORDING' : 'YOUR HAND'}</span><div className="hand-lane">{tiles(hand)}{!hand.length && <span className="zone-empty">No visible hand cards in this position</span>}</div></div>
      <details className="other-zones"><summary>Graveyards, exile & known information <span>{other.length} cards</span></summary><div>{other.map(card => <button key={card.ref} onClick={() => inspect(card)}>{card.name}<small>{card.zone}</small></button>)}{view.known.map((text, index) => <p key={index}>{text}</p>)}</div></details>
      <footer className="transport"><div className="transport-buttons"><button aria-label="First position" disabled={position === 0} onClick={() => move(0)}>⇤</button><button aria-label="Previous position" disabled={position === 0} onClick={() => move(position - 1)}>←</button><button className="play" aria-label={playing ? 'Pause replay' : 'Play replay'} disabled={position === replay.frames.length - 1 && !playing} onClick={togglePlay}>{playing ? 'Ⅱ' : '▶'}</button><button aria-label="Next position" disabled={position === replay.frames.length - 1} onClick={() => move(position + 1)}>→</button><button aria-label="Next decision" disabled={position === replay.frames.length - 1} onClick={() => move(nextPosition(replay, position, 1, true))}>⇥</button></div><input aria-label="Replay position" type="range" min={0} max={replay.frames.length - 1} value={position} onChange={event => move(Number(event.target.value))} /><span className="position-count">{position + 1} / {replay.frames.length}</span></footer>
    </section>
    <aside className="analysis-column" aria-label="Analysis panel">
      <section className="analysis-box"><div className="section-title"><h2>Analysis</h2><span className="model-chip">JEV</span></div>
        <div className="analysis-state" role="status">{sample ? <><strong>A place to test your instincts.</strong><p>This is an interface example. Import your own game to ask Jev.</p></> : !decision ? <><strong>No captured decision</strong><p>Step to a priority point to compare legal choices.</p></> : !decision.supported ? <><strong>Not analyzable</strong><p>{decision.reason}</p></> : result?.status === 'pending' ? <><strong>Analyzing the position…</strong><p>You can keep stepping.</p></> : result?.status === 'error' ? <><strong>Analysis unavailable</strong><p>{result.error}</p></> : result?.status === 'done' ? <><strong>{decision.options.find(option => option.id === result.choice)?.label}</strong><p>{decision.chosen?.includes(result.choice!) ? 'Matches your recorded choice.' : 'Different from your recorded choice.'}</p></> : <><strong>{decision.options.length} captured choices</strong><p>{saved ? 'Choose Analyze to review this position.' : 'Save this replay to use analysis.'}</p></>}</div>
        <div className="analysis-actions">{!hasKey && eligible ? <button className="primary" onClick={settings}>Connect OpenRouter</button> : <button className="primary" disabled={!eligible || !hasKey || result?.status === 'pending' || result?.status === 'done'} onClick={analyze}><StableText value={result?.status === 'pending' ? 'Analyzing…' : result?.status === 'error' ? 'Retry analysis' : 'Analyze'} candidates={['Analyzing…', 'Retry analysis', 'Analyze']} /></button>}<label className="auto-control"><input type="checkbox" checked={auto} disabled={!hasKey} onChange={event => setAuto(event.target.checked)} /> After stepping</label></div>
        <div className="options-list">{options.map(option => <div className={'option ' + (result?.choice === option.id ? 'recommended' : '')} key={option.id}><div><strong title={option.detail}>{option.label}</strong>{decision?.chosen?.includes(option.id) && <span className="played-chip">Played</span>}</div><span>{result?.probabilities?.[option.id] !== undefined ? (result.probabilities[option.id] * 100).toFixed(1) + '%' : '—'}</span></div>)}</div>
        <div className="analysis-metrics" title={result ? `Input ${result.input ?? 'unknown'} · Output ${result.output ?? 'unknown'}` : undefined}><span>{result?.latency != null ? result.latency + ' ms' : 'Latency —'}</span><span>{result?.cost != null ? '$' + result.cost.toFixed(6) : 'Cost —'}</span></div>
        <label className="rating">Your review <select aria-label="Rate analysis" disabled={!saved || result?.status !== 'done' || busy} value={rating} onChange={event => rate(event.target.value)}><option value="unrated">Not rated</option><option value="useful">Useful</option><option value="wrong">Wrong</option><option value="unsure">Unsure</option></select></label><p className="score-note">Choice scores, not win probabilities.</p>
      </section>
      <section className="move-history"><div className="section-title"><h2>Timeline</h2><span>{replay.frames.length} positions</span></div><ol ref={history}>{replay.frames.map((item, index) => <li key={index}><button aria-current={index === position ? 'step' : undefined} onClick={() => move(index)}><span>{index + 1}</span><span>{item.label}</span><small>T{item.turn}</small></button></li>)}</ol></section>
      <details className="capture-notes"><summary>Recording details</summary>{replay.warnings.concat(frame.warnings).map((warning, index) => <p key={index}>{warning}</p>)}</details>
    </aside>
  </main>;
}
