import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  buildUpstreamBody,
  emitClientJson,
  emitClientSse,
  openAIError,
  openAIModelList,
  parseClientRequest,
  parseUpstreamAnswer,
  type NormalizedAnswer,
  type NormalizedRequest,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

interface ProviderState {
  calls: number;
  transientFails: number;
  lastCallFailed: boolean;
  dead: boolean;
}

const providerStates = new Map<string, ProviderState>();
let totalRefusals = 0;
let lossSoFar = 0;

const BUDGET_META: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};

const SAFE_QUOTA: Record<string, number> = {
  openrouter: 1,
  zai: 1,
  cloudflare: 1,
  sambanova: 2,
  cerebras: 2,
  gemini: 3,
  ovh: 2,
  groq: 5,
  kilol: 3,
  pollinations: 0,
};

function stateOf(providerId: string): ProviderState {
  let st = providerStates.get(providerId);
  if (st === undefined) {
    st = { calls: 0, transientFails: 0, lastCallFailed: false, dead: false };
    providerStates.set(providerId, st);
  }
  return st;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS, brief.ratings)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  loss: number;
}

function bestRatingForRequest(reqId: string, requires: Requirement): number {
  const fromBest = brief.ratings?.best.find((b) => b.id === reqId)?.rating_milli;
  if (fromBest !== undefined && fromBest > 0) return fromBest;
  let best = 1475;
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      best = Math.max(best, row?.rating_milli ?? 0);
    }
  }
  return best;
}

function rankCandidates(
  requires: Requirement,
  parsed: NormalizedRequest,
  idx: number,
  manifestEntry: WorkloadRequest | undefined,
): { candidates: Candidate[]; topRating: number; weight: number } {
  const manifest = brief.workload?.manifest ?? [];
  const weight = manifestEntry?.weight ?? 1;
  const reqId = manifestEntry?.id ?? `r${String(idx + 1).padStart(2, '0')}`;
  const topRating = bestRatingForRequest(reqId, requires);

  const future = manifest.slice(idx + 1);
  const futureToolsVision = future.filter((r) => r.requires.tools === true && r.requires.vision === true).length;
  const futureVisionOnly = future.filter((r) => r.requires.vision === true && r.requires.tools !== true).length;
  const future200k = future.filter((r) => (r.requires.min_context ?? 0) >= 200000).length;
  const futureJson = future.filter((r) => r.requires.json === true).length;

  const isToolsVision = requires.tools === true && requires.vision === true;
  const isVisionOnly = requires.vision === true && requires.tools !== true;
  const is200k = (requires.min_context ?? 0) >= 200000;
  const isJson = requires.json === true || parsed.jsonMode;

  const candidates: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const st = stateOf(probe.provider.id);
    if (st.dead) continue;
    // Anthropic upstream wire cannot echo json_mode: true when OpenAI client requested response_format
    if (parsed.jsonMode && probe.provider.wire === 'anthropic') continue;

    let bestModel: ProbedModel | null = null;
    let bestRating = -1;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 1000;
      if (rating > bestRating) {
        bestRating = rating;
        bestModel = model;
      }
    }
    if (bestModel !== null) {
      const credit = Math.min(bestRating, topRating);
      const loss = weight * (topRating - credit);
      candidates.push({ probe, model: bestModel, rating: bestRating, loss });
    }
  }

  candidates.sort((a, b) => {
    const scoreOf = (c: Candidate): number => {
      const pid = c.probe.provider.id;
      const st = stateOf(pid);
      const calls = st.calls;
      const meta = BUDGET_META[pid] ?? { min: 5, max: 6 };
      const minFutureLoss = future200k * 17 + futureJson * 10;
      const exceedsQualityFloor = lossSoFar + c.loss + minFutureLoss > 745;
      const hardExceedsFloor = lossSoFar + c.loss > 760;

      let reserved = false;
      if (!isToolsVision && futureToolsVision > 0 && (pid === 'gemini' || pid === 'zai')) reserved = true;
      if (!isVisionOnly && !isToolsVision && futureVisionOnly > 0 && pid === 'sambanova' && calls === 0) reserved = true;

      if (totalRefusals === 0) {
        if (!is200k && !isToolsVision && future200k > 0) {
          if (pid === 'ovh' && calls < 2) reserved = true;
          if (pid === 'gemini' && calls < 3) reserved = true;
          if ((pid === 'openrouter' || pid === 'cloudflare' || pid === 'zai') && calls === 0 && weight < 2) reserved = true;
        }
        if (!isJson && !is200k && futureJson > 0 && weight < 2) {
          if ((pid === 'sambanova' || pid === 'cerebras') && calls < 2) reserved = true;
        }
        const quota = SAFE_QUOTA[pid] ?? 1;
        const withinQuota = calls < quota && st.transientFails === 0;
        let s = 0;
        if (!hardExceedsFloor) s += 50000;
        if (!exceedsQualityFloor) s += 10000;
        if (!reserved && withinQuota) s += 5000;
        if (calls === 0 && !reserved) s += 1000;
        s += c.rating - st.transientFails * 150 - Math.max(0, calls - quota) * 200;
        return s;
      }

      if (!is200k && !isToolsVision && future200k > 0) {
        if ((pid === 'ovh' || pid === 'zai' || pid === 'cloudflare') && calls === 0) reserved = true;
      }
      const avgB = (meta.min + meta.max) / 2;
      let failProb = 0;
      if (calls === 0) {
        failProb = 0;
      } else if (calls >= meta.max) {
        failProb = 0.95;
      } else if (st.transientFails >= 2 && calls < meta.min) {
        failProb = 0.01;
      } else {
        const remFails = Math.max(0.15, 1.5 - 0.85 * st.transientFails);
        const remSlots = Math.max(1.0, avgB - 1 - calls);
        failProb = Math.min(0.85, remFails / remSlots);
        if (calls >= meta.min) failProb = Math.min(0.9, failProb + 0.35);
        if (st.lastCallFailed && st.transientFails === 1) failProb += 0.1;
      }
      let s = 0;
      if (!hardExceedsFloor) s += 50000;
      if (!exceedsQualityFloor) s += 10000;
      if (!reserved) s += 3000;
      s += c.rating * 1.2 - failProb * 1800;
      return s;
    };
    return scoreOf(b) - scoreOf(a);
  });

  return { candidates, topRating, weight };
}

function mergeRequirements(fromManifest: Requirement | undefined, parsed: NormalizedRequest): Requirement {
  return {
    tools: fromManifest?.tools === true || parsed.tools.length > 0,
    vision: fromManifest?.vision === true || parsed.images.length > 0,
    json: fromManifest?.json === true || parsed.jsonMode,
    ...(fromManifest?.min_context !== undefined ? { min_context: fromManifest.min_context } : {}),
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, payload: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(payload);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function callUpstream(cand: Candidate, req: NormalizedRequest): Promise<{ ok: boolean; answer: NormalizedAnswer | null; permanentDead: boolean }> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const url =
    provider.wire === 'openai'
      ? `${base}/v1/chat/completions`
      : provider.wire === 'anthropic'
        ? `${base}/v1/messages`
        : `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:generateContent`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const upstreamBody = buildUpstreamBody(provider.wire, cand.model.id, req);

  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  const rawText = await res.text();
  if (res.ok) {
    let parsedJson: unknown = {};
    try {
      parsedJson = JSON.parse(rawText);
    } catch {
      parsedJson = {};
    }
    const answer = parseUpstreamAnswer(provider.wire, isRecord(parsedJson) ? parsedJson : {});
    return { ok: true, answer, permanentDead: false };
  }

  // Distinguish permanent exhaustion/outage ("budget is spent" / "is out") from single-call fail_calls ("refused this call")
  const permanentDead =
    res.status === 401 ||
    res.status === 403 ||
    rawText.includes('budget is spent') ||
    rawText.includes('is out');
  return { ok: false, answer: null, permanentDead };
}

async function serveInference(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');

  const idx = manifestIndex;
  manifestIndex += 1;
  const manifestEntry = brief.workload?.manifest[idx];
  const parsed = parseClientRequest(clientApi, body);
  const requires = mergeRequirements(manifestEntry?.requires, parsed);
  const { candidates, topRating, weight } = rankCandidates(requires, parsed, idx, manifestEntry);

  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, 'overloaded_error', 'no free model on the directory meets this request');
  }

  for (let pass = 0; pass < 3; pass++) {
    for (const cand of candidates) {
      const st = stateOf(cand.probe.provider.id);
      if (st.dead) continue;
      st.calls += 1;
      try {
        const outcome = await callUpstream(cand, parsed);
        if (outcome.ok && outcome.answer !== null) {
          st.lastCallFailed = false;
          const credit = Math.min(cand.rating, topRating);
          lossSoFar += weight * (topRating - credit);
          if (parsed.stream) {
            sendSse(res, emitClientSse(clientApi, outcome.answer, cand.model.id));
          } else {
            sendJson(res, 200, emitClientJson(clientApi, outcome.answer, cand.model.id));
          }
          return;
        }
        totalRefusals += 1;
        st.lastCallFailed = true;
        if (outcome.permanentDead) {
          st.dead = true;
        } else {
          st.transientFails += 1;
        }
      } catch {
        totalRefusals += 1;
        st.lastCallFailed = true;
        st.transientFails += 1;
      }
    }
  }

  sendError(res, clientApi, 503, 'overloaded_error', 'all candidate providers refused');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveInference(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveInference(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry on next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
