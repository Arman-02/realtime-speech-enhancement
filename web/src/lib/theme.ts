import { useCallback, useEffect, useState } from 'react';

export type Theme = 'dark' | 'light';

function current(): Theme {
  return document.documentElement.classList.contains('light') ? 'light' : 'dark';
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(current);
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('light', theme === 'light');
    root.classList.toggle('dark', theme === 'dark');
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#f7f7f5' : '#0a0c10');
    try {
      localStorage.setItem('ans-theme', theme);
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
    window.dispatchEvent(new CustomEvent('ans-theme', { detail: theme }));
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);
  return [theme, toggle];
}

/** Re-render when the theme changes (for canvas drawings that read CSS variables). */
export function useThemeVersion(): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const on = () => setV((x) => x + 1);
    window.addEventListener('ans-theme', on);
    return () => window.removeEventListener('ans-theme', on);
  }, []);
  return v;
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
