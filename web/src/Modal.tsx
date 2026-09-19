import { useEffect, useRef, type ReactNode } from 'react';

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    const origin = document.activeElement;
    dialog.showModal();
    return () => {
      dialog.close();
      if (origin instanceof HTMLElement && origin.isConnected) origin.focus();
    };
  }, []);
  return <dialog ref={ref} aria-label={title} onCancel={e => { e.preventDefault(); onClose(); }} onClick={e => {
    if (e.target !== ref.current || !ref.current) return;
    const box = ref.current.getBoundingClientRect();
    if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) onClose();
  }}>
    <header className="dialog-head"><h2>{title}</h2><button className="icon" onClick={onClose} aria-label="Close dialog">×</button></header>
    {children}
  </dialog>;
}
