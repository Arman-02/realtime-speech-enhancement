/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_REPO_URL?: string;
  /** Optional form endpoint (Formspree, a Google Apps Script, ...) that receives listening-test responses as JSON. */
  readonly VITE_LISTEN_ENDPOINT?: string;
  /** "1" = dev/test build that reads src/fixtures/* instead of public/ data files. Never set for a real deploy. */
  readonly VITE_USE_FIXTURES?: string;
}
