import { lazy, Suspense, useEffect } from 'react';
import { Layout } from './components/Layout';
import { useRoute, type Route } from './lib/router';
import Studio from './pages/Studio';

const Live = lazy(() => import('./pages/Live'));
const Results = lazy(() => import('./pages/Results'));
const HowItWorks = lazy(() => import('./pages/HowItWorks'));
const Diagnostics = lazy(() => import('./pages/Diagnostics'));
const Listen = lazy(() => import('./pages/Listen'));

const TITLES: Record<Route, string> = {
  studio: 'Studio',
  live: 'Live',
  results: 'Results',
  how: 'How it works',
  diagnostics: 'Diagnostics',
  listen: 'Listening test',
  notfound: 'Not found',
};

function Page({ route }: { route: Route }) {
  switch (route) {
    case 'studio':
      return <Studio />;
    case 'live':
      return <Live />;
    case 'results':
      return <Results />;
    case 'how':
      return <HowItWorks />;
    case 'diagnostics':
      return <Diagnostics />;
    case 'listen':
      return <Listen />;
    default:
      return (
        <div className="py-24 text-center">
          <h1 className="text-2xl font-semibold">Page not found</h1>
          <p className="mt-2 text-fg-2">
            <a className="text-accent underline-offset-2 hover:underline" href="#/">
              Back to the Studio
            </a>
          </p>
        </div>
      );
  }
}

function Fallback() {
  return (
    <div className="flex items-center gap-3 py-24 text-sm text-fg-3" role="status">
      <span className="pulse-dot h-2 w-2 rounded-full bg-accent" /> Loading…
    </div>
  );
}

export default function App() {
  const route = useRoute();
  useEffect(() => {
    document.title = `${TITLES[route]} · Real-Time Speech Enhancement`;
  }, [route]);
  return (
    <Layout route={route}>
      <Suspense fallback={<Fallback />}>
        <div key={route} className="fade-up">
          <Page route={route} />
        </div>
      </Suspense>
    </Layout>
  );
}
