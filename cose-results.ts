// Shared analysis-results store for the CoSE image ecosystem.
// KEEP IDENTICAL across the AstroBotany database, AstroRoot, and this app — same
// origin (dr-richard-barker.github.io) means they share this IndexedDB store, so
// results written here are read back by the database. See the database repo's
// src/lib/cose-results.ts.

const DB_NAME = 'cose-analysis';
const STORE = 'results';

export interface AnalysisResult {
  id: string;            // `${ref}::${tool}`
  ref: string;
  imageUrl: string;
  tool: string;
  toolName: string;
  metrics: Record<string, string | number>;
  generatedAt: string;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function putResult(r: Omit<AnalysisResult, 'id'>): Promise<void> {
  const db = await open();
  const rec: AnalysisResult = { ...r, id: `${r.ref}::${r.tool}` };
  await new Promise<void>((resolve, reject) => {
    const t = db.transaction(STORE, 'readwrite');
    t.objectStore(STORE).put(rec);
    t.oncomplete = () => { db.close(); resolve(); };
    t.onerror = () => reject(t.error);
  });
}

// The ref the database handed off via ?ref= (falls back to the image URL).
export function currentRef(imageUrl?: string): string {
  return new URLSearchParams(location.search).get('ref') || imageUrl || 'unlinked';
}
