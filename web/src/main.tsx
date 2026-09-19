import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles.css';
class ErrorBoundary extends React.Component<{children:React.ReactNode},{failed:boolean}>{
  state={failed:false};
  static getDerivedStateFromError(){return {failed:true}}
  render(){return this.state.failed?<main className="fatal"><h1>This view could not open.</h1><p>Your saved replays are unchanged.</p><button onClick={()=>location.reload()}>Reload</button></main>:this.props.children}
}
createRoot(document.getElementById('root')!).render(<ErrorBoundary><App/></ErrorBoundary>);
