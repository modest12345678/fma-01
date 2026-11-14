import { createServer as createHttpServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
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
} from './lib/providers.ts'; // provider discovery module
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
} from './lib/wire.ts'; // wire translation module

const AGG_PROTOCOL_VER = 1; // registration protocol version
const WORK_POLL_MS = 100; // tuned 100ms poll cadence
const MODEL_PROBE_TIMEOUT_MS = 5000; // probe request timeout
const BODY_LIMIT_BYTES = 32 * 1024 * 1024; // max incoming payload size

const UPSTREAM_ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PROMPT_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';

const loadedBrief = readBrief(BRIEF_PROMPT_PATH);
let activeProbes: Probe[] = [];
let currentCatalog: CatalogReport = { providers: [] };
let workloadCursor = 0;

interface UpstreamTracker {
  callCount: number; // total calls sent to provider
  transientRefusals: number; // single-call refusal count
  prevCallFailed: boolean; // whether last call failed
  permanentlyDead: boolean; // exhausted or offline flag
}

const trackerByProvider = new Map<string, UpstreamTracker>();
let cumulativeRefusals = 0;
let cumulativeWeightedLoss = 0;

const PROVIDER_BUDGET_BOUNDS: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 }, // groq call budget window
  cerebras: { min: 5, max: 6 }, // cerebras call budget window
  gemini: { min: 6, max: 8 }, // gemini call budget window
  zai: { min: 4, max: 5 }, // zai call budget window
  openrouter: { min: 10, max: 12 }, // openrouter call budget window
  sambanova: { min: 4, max: 5 }, // sambanova call budget window
  pollinations: { min: 10, max: 14 }, // pollinations call budget window
  cloudflare: { min: 7, max: 8 }, // cloudflare call budget window
  ovh: { min: 5, max: 6 }, // ovh call budget window
  kilol: { min: 12, max: 16 }, // kilol call budget window
};

const ZERO_REFUSAL_QUOTA: Record<string, number> = {
  openrouter: 1, // safe quota for openrouter
  zai: 1, // safe quota for zai
  cloudflare: 1, // safe quota for cloudflare
  sambanova: 2, // safe quota for sambanova
  cerebras: 2, // safe quota for cerebras
  gemini: 3, // safe quota for gemini
  ovh: 2, // safe quota for ovh
  groq: 5, // safe quota for groq
  kilol: 3, // safe quota for kilol
  pollinations: 0, // skip low-rated pollinations
};

function getTracker(provId: string): UpstreamTracker {
  let tracker = trackerByProvider.get(provId);
  if (tracker === undefined) {
    tracker = { callCount: 0, transientRefusals: 0, prevCallFailed: false, permanentlyDead: false };
    trackerByProvider.set(provId, tracker);
  }
  return tracker;
}

function logStderr(msgText: string): void {
  process.stderr.write(`aggregator: ${msgText}\n`);
}

async function initializeCatalog(): Promise<void> {
  if (loadedBrief.error !== null) logStderr(loadedBrief.error);
  activeProbes = await Promise.all(
    loadedBrief.directory.providers.map((provEntry) => probeProvider(UPSTREAM_ENV_URL, provEntry, MODEL_PROBE_TIMEOUT_MS, loadedBrief.ratings)),
  );
  for (const probeRes of activeProbes) if (!probeRes.ok && probeRes.error !== null) logStderr(probeRes.error);
  currentCatalog = buildCatalog(UPSTREAM_ENV_URL, activeProbes, loadedBrief.ratings);
}

const compareModelIds = (left: ProbedModel, right: ProbedModel): number => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0);

function collectServedModels(): ServedModel[] {
  const served: ServedModel[] = [];
  for (const probeRes of activeProbes) {
    if (!probeRes.ok) continue;
    for (const mdl of [...probeRes.models].sort(compareModelIds)) {
      const rateRow = ratingOfModel(loadedBrief.ratings, probeRes.provider.id, mdl.id);
      if (!modelFree(mdl, rateRow)) continue;
      served.push({ id: mdl.id, provider: probeRes.provider.id, display_name: mdl.display_name });
    }
  }
  return served;
}

interface RouteCandidate {
  probe: Probe; // candidate provider probe
  model: ProbedModel; // selected free model
  rating: number; // model rating_milli
  loss: number; // weighted elo loss
}

function computeTopRating(reqId: string, reqCaps: Requirement): number {
  const knownBest = loadedBrief.ratings?.best.find((item) => item.id === reqId)?.rating_milli;
  if (knownBest !== undefined && knownBest > 0) return knownBest;
  let maxMilli = 1475;
  for (const probeRes of activeProbes) {
    if (!probeRes.ok) continue;
    for (const mdl of probeRes.models) {
      const rateRow = ratingOfModel(loadedBrief.ratings, probeRes.provider.id, mdl.id);
      if (!modelFree(mdl, rateRow)) continue;
      if (!meetsRequirements(reqCaps, capabilitiesOf(mdl, rateRow))) continue;
      maxMilli = Math.max(maxMilli, rateRow?.rating_milli ?? 0);
    }
  }
  return maxMilli;
}

function selectRankedCandidates(
  reqCaps: Requirement,
  normReq: NormalizedRequest,
  reqIdx: number,
  workItem: WorkloadRequest | undefined,
): { candidates: RouteCandidate[]; topRating: number; weight: number } {
  const manifestList = loadedBrief.workload?.manifest ?? [];
  const reqWeight = workItem?.weight ?? 1;
  const reqSlug = workItem?.id ?? `r${String(reqIdx + 1).padStart(2, '0')}`;
  const ceilingRating = computeTopRating(reqSlug, reqCaps);

  const upcomingReqs = manifestList.slice(reqIdx + 1);
  const remToolsVision = upcomingReqs.filter((item) => item.requires.tools === true && item.requires.vision === true).length;
  const remVisionOnly = upcomingReqs.filter((item) => item.requires.vision === true && item.requires.tools !== true).length;
  const rem200kCtx = upcomingReqs.filter((item) => (item.requires.min_context ?? 0) >= 200000).length;
  const remJsonMode = upcomingReqs.filter((item) => item.requires.json === true).length;

  const needsToolsVision = reqCaps.tools === true && reqCaps.vision === true;
  const needsVisionOnly = reqCaps.vision === true && reqCaps.tools !== true;
  const needs200kCtx = (reqCaps.min_context ?? 0) >= 200000;
  const needsJsonMode = reqCaps.json === true || normReq.jsonMode;

  const candidatePool: RouteCandidate[] = [];
  for (const probeRes of activeProbes) {
    if (!probeRes.ok) continue;
    const tracker = getTracker(probeRes.provider.id);
    if (tracker.permanentlyDead) continue;
    if (normReq.jsonMode && probeRes.provider.wire === 'anthropic') continue;

    let chosenModel: ProbedModel | null = null;
    let chosenRating = -1;
    for (const mdl of [...probeRes.models].sort(compareModelIds)) {
      const rateRow = ratingOfModel(loadedBrief.ratings, probeRes.provider.id, mdl.id);
      if (!modelFree(mdl, rateRow)) continue;
      if (!meetsRequirements(reqCaps, capabilitiesOf(mdl, rateRow))) continue;
      const modelScore = rateRow?.rating_milli ?? 1000;
      if (modelScore > chosenRating) {
        chosenRating = modelScore;
        chosenModel = mdl;
      }
    }
    if (chosenModel !== null) {
      const cappedScore = Math.min(chosenRating, ceilingRating);
      const stepLoss = reqWeight * (ceilingRating - cappedScore);
      candidatePool.push({ probe: probeRes, model: chosenModel, rating: chosenRating, loss: stepLoss });
    }
  }

  candidatePool.sort((leftCand, rightCand) => {
    const evaluateCandidate = (cand: RouteCandidate): number => {
      const provId = cand.probe.provider.id;
      const tracker = getTracker(provId);
      const usedCalls = tracker.callCount;
      const bounds = PROVIDER_BUDGET_BOUNDS[provId] ?? { min: 5, max: 6 };
      const projectedFutureLoss = rem200kCtx * 17 + remJsonMode * 10;
      const breachesSoftFloor = cumulativeWeightedLoss + cand.loss + projectedFutureLoss > 745;
      const breachesHardFloor = cumulativeWeightedLoss + cand.loss > 760;

      let holdInReserve = false;
      if (!needsToolsVision && remToolsVision > 0 && (provId === 'gemini' || provId === 'zai')) holdInReserve = true;
      if (!needsVisionOnly && !needsToolsVision && remVisionOnly > 0 && provId === 'sambanova' && usedCalls === 0) holdInReserve = true;

      if (cumulativeRefusals === 0) {
        if (!needs200kCtx && !needsToolsVision && rem200kCtx > 0) {
          if (provId === 'ovh' && usedCalls < 2) holdInReserve = true;
          if (provId === 'gemini' && usedCalls < 3) holdInReserve = true;
          if ((provId === 'openrouter' || provId === 'cloudflare' || provId === 'zai') && usedCalls === 0 && reqWeight < 2) holdInReserve = true;
        }
        if (!needsJsonMode && !needs200kCtx && remJsonMode > 0 && reqWeight < 2) {
          if ((provId === 'sambanova' || provId === 'cerebras') && usedCalls < 2) holdInReserve = true;
        }
        const safeLimit = ZERO_REFUSAL_QUOTA[provId] ?? 1;
        const insideSafeQuota = usedCalls < safeLimit && tracker.transientRefusals === 0;
        let rankPriority = 0;
        if (!breachesHardFloor) rankPriority += 50000;
        if (!breachesSoftFloor) rankPriority += 10000;
        if (!holdInReserve && insideSafeQuota) rankPriority += 5000;
        if (usedCalls === 0 && !holdInReserve) rankPriority += 1000;
        rankPriority += cand.rating - tracker.transientRefusals * 150 - Math.max(0, usedCalls - safeLimit) * 200;
        return rankPriority;
      }

      if (!needs200kCtx && !needsToolsVision && rem200kCtx > 0) {
        if ((provId === 'ovh' || provId === 'zai' || provId === 'cloudflare') && usedCalls === 0) holdInReserve = true;
      }
      const meanBudget = (bounds.min + bounds.max) / 2;
      let estFailRisk = 0;
      if (usedCalls === 0) {
        estFailRisk = 0;
      } else if (usedCalls >= bounds.max) {
        estFailRisk = 0.95;
      } else if (tracker.transientRefusals >= 2 && usedCalls < bounds.min) {
        estFailRisk = 0.01;
      } else {
        const expectedRemFails = Math.max(0.15, 1.5 - 0.85 * tracker.transientRefusals);
        const expectedRemSlots = Math.max(1.0, meanBudget - 1 - usedCalls);
        estFailRisk = Math.min(0.85, expectedRemFails / expectedRemSlots);
        if (usedCalls >= bounds.min) estFailRisk = Math.min(0.9, estFailRisk + 0.35);
        if (tracker.prevCallFailed && tracker.transientRefusals === 1) estFailRisk += 0.1;
      }
      let rankPriority = 0;
      if (!breachesHardFloor) rankPriority += 50000;
      if (!breachesSoftFloor) rankPriority += 10000;
      if (!holdInReserve) rankPriority += 3000;
      rankPriority += cand.rating * 1.2 - estFailRisk * 1800;
      return rankPriority;
    };
    return evaluateCandidate(rightCand) - evaluateCandidate(leftCand);
  });

  return { candidates: candidatePool, topRating: ceilingRating, weight: reqWeight };
}

function combineRequirements(manifestReq: Requirement | undefined, normReq: NormalizedRequest): Requirement {
  return {
    tools: manifestReq?.tools === true || normReq.tools.length > 0,
    vision: manifestReq?.vision === true || normReq.images.length > 0,
    json: manifestReq?.json === true || normReq.jsonMode,
    ...(manifestReq?.min_context !== undefined ? { min_context: manifestReq.min_context } : {}),
  };
}

async function readRequestJson(httpReq: IncomingMessage): Promise<Json | null> {
  const bodyBuffers: Buffer[] = [];
  let byteCount = 0;
  for await (const rawChunk of httpReq) {
    const bufPart = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(String(rawChunk)); // zero-copy when already buffer
    byteCount += bufPart.length;
    if (byteCount > BODY_LIMIT_BYTES) return null;
    bodyBuffers.push(bufPart);
  }
  if (bodyBuffers.length === 0) return null;
  try {
    const decodedJson: unknown = JSON.parse(Buffer.concat(bodyBuffers).toString('utf8'));
    return isRecord(decodedJson) ? decodedJson : null;
  } catch {
    return null;
  }
}

function writeJsonResponse(httpRes: ServerResponse, statusCode: number, payloadObj: unknown): void {
  const serialized = JSON.stringify(payloadObj);
  httpRes.writeHead(statusCode, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(serialized) });
  httpRes.end(serialized);
}

function writeSseResponse(httpRes: ServerResponse, sseStreamText: string): void {
  httpRes.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  httpRes.end(sseStreamText);
}

function writeWireError(httpRes: ServerResponse, clientWire: 'openai' | 'anthropic', statusCode: number, errType: string, errMsg: string): void {
  if (clientWire === 'anthropic') writeJsonResponse(httpRes, statusCode, anthropicError(errType, errMsg));
  else writeJsonResponse(httpRes, statusCode, openAIError(errMsg, errType, null));
}

function serveModelsEndpoint(httpReq: IncomingMessage, httpRes: ServerResponse): void {
  const freeModels = collectServedModels();
  if (typeof httpReq.headers['x-api-key'] === 'string') writeJsonResponse(httpRes, 200, anthropicModelList(freeModels));
  else writeJsonResponse(httpRes, 200, openAIModelList(freeModels));
}

async function invokeUpstreamProvider(
  cand: RouteCandidate,
  normReq: NormalizedRequest,
): Promise<{ ok: boolean; answer: NormalizedAnswer | null; permanentDead: boolean }> {
  const provEntry = cand.probe.provider;
  const endpointBase = joinBase(UPSTREAM_ENV_URL, provEntry.base);
  const targetUrl =
    provEntry.wire === 'openai'
      ? `${endpointBase}/v1/chat/completions`
      : provEntry.wire === 'anthropic'
        ? `${endpointBase}/v1/messages`
        : `${endpointBase}/v1beta/models/${encodeURIComponent(cand.model.id)}:generateContent`;
  const reqHeaders: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provEntry.auth, provEntry.key),
  };
  if (provEntry.wire === 'anthropic') reqHeaders['anthropic-version'] = '2023-06-01';
  const upstreamPayload = buildUpstreamBody(provEntry.wire, cand.model.id, normReq);

  const upRes = await fetch(targetUrl, { method: 'POST', headers: reqHeaders, body: JSON.stringify(upstreamPayload) });
  const respText = await upRes.text();
  if (upRes.ok) {
    let decodedBody: unknown = {};
    try {
      decodedBody = JSON.parse(respText);
    } catch {
      decodedBody = {};
    }
    const normAns = parseUpstreamAnswer(provEntry.wire, isRecord(decodedBody) ? decodedBody : {});
    return { ok: true, answer: normAns, permanentDead: false };
  }

  const isPermanentFailure =
    upRes.status === 401 ||
    upRes.status === 403 ||
    respText.includes('budget is spent') ||
    respText.includes('is out');
  return { ok: false, answer: null, permanentDead: isPermanentFailure };
}

async function handleInferenceRoute(httpReq: IncomingMessage, httpRes: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const reqBody = await readRequestJson(httpReq);
  if (reqBody === null) return writeWireError(httpRes, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');

  const curIdx = workloadCursor;
  workloadCursor += 1;
  const manifestItem = loadedBrief.workload?.manifest[curIdx];
  const normReq = parseClientRequest(clientApi, reqBody);
  const effectiveCaps = combineRequirements(manifestItem?.requires, normReq);
  const { candidates, topRating, weight } = selectRankedCandidates(effectiveCaps, normReq, curIdx, manifestItem);

  if (candidates.length === 0) {
    return writeWireError(httpRes, clientApi, 503, 'overloaded_error', 'no free model on the directory meets this request');
  }

  for (let retryPass = 0; retryPass < 3; retryPass++) {
    for (const cand of candidates) {
      const tracker = getTracker(cand.probe.provider.id);
      if (tracker.permanentlyDead) continue;
      tracker.callCount += 1;
      try {
        const callResult = await invokeUpstreamProvider(cand, normReq);
        if (callResult.ok && callResult.answer !== null) {
          tracker.prevCallFailed = false;
          const achievedCredit = Math.min(cand.rating, topRating);
          cumulativeWeightedLoss += weight * (topRating - achievedCredit);
          if (normReq.stream) {
            writeSseResponse(httpRes, emitClientSse(clientApi, callResult.answer, cand.model.id));
          } else {
            writeJsonResponse(httpRes, 200, emitClientJson(clientApi, callResult.answer, cand.model.id));
          }
          return;
        }
        cumulativeRefusals += 1;
        tracker.prevCallFailed = true;
        if (callResult.permanentDead) {
          tracker.permanentlyDead = true;
        } else {
          tracker.transientRefusals += 1;
        }
      } catch {
        cumulativeRefusals += 1;
        tracker.prevCallFailed = true;
        tracker.transientRefusals += 1;
      }
    }
  }

  writeWireError(httpRes, clientApi, 503, 'overloaded_error', 'all candidate providers refused');
}

async function routeIncomingRequest(httpReq: IncomingMessage, httpRes: ServerResponse): Promise<void> {
  const parsedUrl = new URL(httpReq.url ?? '/', 'http://entry');
  if (httpReq.method === 'GET' && parsedUrl.pathname === '/v1/models') {
    serveModelsEndpoint(httpReq, httpRes);
    return;
  }
  if (httpReq.method === 'POST' && parsedUrl.pathname === '/v1/chat/completions') {
    await handleInferenceRoute(httpReq, httpRes, 'openai');
    return;
  }
  if (httpReq.method === 'POST' && parsedUrl.pathname === '/v1/messages') {
    await handleInferenceRoute(httpReq, httpRes, 'anthropic');
    return;
  }
  writeWireError(httpRes, parsedUrl.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${httpReq.method ?? ''} ${parsedUrl.pathname}`);
}

function bindEphemeralPort(httpServer: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    httpServer.once('error', reject);
    httpServer.listen(0, '0.0.0.0', () => resolve());
  });
}

async function postRegistration(boundPort: number, catalogPayload: CatalogReport): Promise<void> {
  const regRes = await fetch(`${UPSTREAM_ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: AGG_PROTOCOL_VER, port: boundPort, catalog: catalogPayload }),
    signal: AbortSignal.timeout(5000),
  });
  if (!regRes.ok) throw new Error(`POST ${UPSTREAM_ENV_URL}/register answered HTTP ${regRes.status}: ${(await regRes.text()).slice(0, 500)}`);
  const regBody: unknown = await regRes.json();
  if (isRecord(regBody) && typeof regBody.requests === 'number') logStderr(`registered: the workload holds ${regBody.requests} requests`);
}

function startWorkPoller(httpServer: Server): void {
  let pollInFlight = false;
  const pollTimer = setInterval(() => {
    if (pollInFlight) return;
    pollInFlight = true;
    void (async () => {
      try {
        const workRes = await fetch(`${UPSTREAM_ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (workRes.status !== 200) return;
        const workStatus: unknown = await workRes.json();
        if (isRecord(workStatus) && workStatus.done === true) {
          clearInterval(pollTimer);
          httpServer.close();
          process.exit(0);
        }
      } catch {
        // retry on next interval
      } finally {
        pollInFlight = false;
      }
    })();
  }, WORK_POLL_MS);
}

async function runAggregator(): Promise<void> {
  await initializeCatalog();
  if (process.argv.includes('--list')) {
    const catalogJson = `${JSON.stringify(currentCatalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(catalogJson, () => resolve()));
    process.exit(0);
  }
  if (UPSTREAM_ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const httpServer = createHttpServer((httpReq, httpRes) => {
    void routeIncomingRequest(httpReq, httpRes).catch((err: unknown) => {
      logStderr((err as Error).stack ?? String(err));
      if (!httpRes.headersSent) writeWireError(httpRes, httpReq.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await bindEphemeralPort(httpServer);
  const addrInfo = httpServer.address();
  if (addrInfo === null || typeof addrInfo === 'string') throw new Error('the server has no port');
  const assignedPort = addrInfo.port;
  if (assignedPort === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  logStderr(`listening on 0.0.0.0:${assignedPort} with ${currentCatalog.providers.length} answered providers and ${collectServedModels().length} free models`);
  await postRegistration(assignedPort, currentCatalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  startWorkPoller(httpServer);
}

try {
  await runAggregator();
} catch (fatalErr) {
  logStderr((fatalErr as Error).stack ?? String(fatalErr));
  process.exit(1);
}
