// Keyed fingerprints. A plain sha256 of a contract chunk would let anyone who can read the logs confirm that a
// known document was scanned; an HMAC with a secret key can't be recomputed without the key.

const enc = new TextEncoder();
const keys = new Map<string, Promise<CryptoKey>>();

// Fallback when a secret is not configured (local dev, tests): a random per-isolate key. Fingerprints then can't be
// linked across isolates, which is safe for privacy but weakens the per-visitor quota, so production sets the secrets.
// Created lazily: workerd forbids random values in global scope (deploy fails with code 10021).
let fallback: string | undefined;
const fallbackKey = () =>
  (fallback ??= [...crypto.getRandomValues(new Uint8Array(32))].map((b) => b.toString(16).padStart(2, "0")).join(""));

function key(secret: string | undefined): Promise<CryptoKey> {
  const s = secret || fallbackKey();
  let k = keys.get(s);
  if (!k) {
    k = crypto.subtle.importKey("raw", enc.encode(s), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    keys.set(s, k);
  }
  return k;
}

export async function hmacHex(secret: string | undefined, text: string, hexChars = 16): Promise<string> {
  const sig = await crypto.subtle.sign("HMAC", await key(secret), enc.encode(text));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, hexChars);
}

/**
 * Pseudonymous visitor id for quotas: HMAC(HMAC(IP_SALT_KEY, day), ip). A new salt every UTC day, so ids can't be
 * linked across days, and the Durable Object deletes the previous day's rows.
 */
export async function visitorId(secret: string | undefined, ip: string, day: string): Promise<string> {
  const salt = await hmacHex(secret, `visitor-salt:${day}`, 64);
  return hmacHex(salt, ip);
}
