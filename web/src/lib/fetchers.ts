/**
 * Fetch helpers that treat "missing" robustly. Dev servers and static hosts with SPA fallbacks often
 * answer a missing file with 200 + index.html, so we sniff the payload rather than trusting the status.
 */

export async function fetchJsonOrNull<T>(url: string, init?: RequestInit): Promise<T | null> {
  let res: Response;
  try {
    res = await fetch(url, { cache: 'no-cache', ...init });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const text = await res.text();
  const head = text.trimStart().slice(0, 1);
  if (head !== '{' && head !== '[') return null;
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

export async function fetchBytesOrNull(url: string, init?: RequestInit): Promise<Uint8Array | null> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const type = res.headers.get('content-type') ?? '';
  if (type.includes('text/html')) return null;
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf.length === 0) return null;
  // An HTML fallback served with a generic content type.
  const sniff = String.fromCharCode(...buf.subarray(0, 16)).trimStart().toLowerCase();
  if (sniff.startsWith('<!doctype') || sniff.startsWith('<html')) return null;
  return buf;
}

/** Absolute URL for a file under the app's base path (works with BASE_PATH sub-path deploys). */
export function assetUrl(path: string, base: string = import.meta.env.BASE_URL): string {
  const root = typeof location !== 'undefined' ? new URL(base, location.href) : new URL(base, 'http://localhost/');
  return new URL(path.replace(/^\//, ''), root).href;
}
