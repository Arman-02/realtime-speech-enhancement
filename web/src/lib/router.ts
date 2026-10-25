import { useEffect, useState } from 'react';

/** Hash routes work on any static host (no rewrites needed). */
export type Route = 'studio' | 'live' | 'results' | 'how' | 'diagnostics' | 'listen' | 'notfound';

const MAP: Record<string, Route> = {
  '': 'studio',
  '/': 'studio',
  '/studio': 'studio',
  '/live': 'live',
  '/results': 'results',
  '/how-it-works': 'how',
  '/how': 'how',
  '/diagnostics': 'diagnostics',
  '/listen': 'listen',
};

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, '').split('?')[0].replace(/\/+$/, '') || '/';
  return MAP[path] ?? 'notfound';
}

export function hrefFor(route: Route): string {
  switch (route) {
    case 'studio':
      return '#/';
    case 'how':
      return '#/how-it-works';
    default:
      return `#/${route}`;
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parseHash(location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}
