import type { View } from './domain';
import { PlayerBar as HeroPlayerBar } from './components';

export function PlayerBar({ view, seat }: { view: View; seat: string }) {
  if (view.viewer !== 'public') return <HeroPlayerBar view={view} seat={seat} />;
  const index = view.players.findIndex(player => player.id === seat);
  const player = view.players[index];
  return <div className="player-bar"><span className="avatar">{index + 1}</span><strong>Player {index + 1}</strong><span className="muted">{player?.handCount ?? '—'} in hand · {player?.libraryCount ?? '—'} in library</span><span className="priority-chip">{view.active === seat ? 'Active turn' : ''}</span><b className="life">{player?.life ?? '—'}<small>life</small></b></div>;
}
