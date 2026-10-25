/**
 * Dev/test-only fixture switch. With `VITE_USE_FIXTURES=1` (dev server or a separate test build) the
 * data loaders read `src/fixtures/*.json` instead of the files under `public/`. Vite replaces the flag
 * statically, so in a normal build the fixture imports are dead code and the JSON is never bundled.
 */
import { assetUrl, fetchJsonOrNull } from './fetchers';

export const USE_FIXTURES = import.meta.env.VITE_USE_FIXTURES === '1';

type FixtureName = 'results' | 'failures' | 'listening' | 'manifest';

async function importFixture(name: FixtureName): Promise<unknown> {
  switch (name) {
    case 'results':
      return (await import('../fixtures/results_v02.json')).default;
    case 'failures':
      return (await import('../fixtures/failures_v1.json')).default;
    case 'listening':
      return (await import('../fixtures/listening_v1.json')).default;
    case 'manifest':
      return (await import('../fixtures/manifest_v02.json')).default;
  }
}

/**
 * Fetch a JSON data file from the site root (`path`, e.g. "listening/index.json"), or the matching
 * fixture (if one is named) when fixtures are enabled. Resolves to null when the file is missing or not JSON.
 */
export async function loadDataFile<T>(path: string, fixture?: FixtureName): Promise<T | null> {
  if (USE_FIXTURES && fixture) return (await importFixture(fixture)) as T;
  return fetchJsonOrNull<T>(assetUrl(path));
}
