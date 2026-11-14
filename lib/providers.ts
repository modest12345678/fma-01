import { readFileSync as readUtf8File } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini'; // upstream wire protocol
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none'; // upstream auth header kind

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1'; // directory schema id
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1'; // ratings schema id
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1'; // workload schema id

export type Json = Record<string, unknown>; // generic json object record
export const isRecord = (val: unknown): val is Json => val !== null && typeof val === 'object' && !Array.isArray(val);

export interface ProviderDirectoryEntry {
  id: string; // provider identifier
  name: string; // display label
  wire: Wire; // protocol kind
  auth: AuthScheme; // credential scheme
  base: string; // endpoint base path
  key_env: string; // env variable name
  key: string; // resolved api token
}

export interface ProviderDirectory {
  schema: string; // directory document schema
  providers: ProviderDirectoryEntry[]; // parsed provider list
}

export interface CatalogModel {
  id: string; // model slug
  free: boolean; // zero-cost flag
  context_length: number | null; // token window size
}

export interface CatalogProvider {
  id: string; // catalog provider id
  base_url: string; // catalog base url
  key_env: string; // catalog key env
  wire: Wire; // catalog wire type
  models: CatalogModel[]; // discovered models
}

export interface CatalogReport {
  providers: CatalogProvider[]; // full catalog payload
}

export interface RatingRow {
  model: string; // rated model id
  provider: string; // rated provider id
  rating_milli: number; // milli-elo score
  context_length: number | null; // rated token window
  tools: boolean; // supports tool use
  vision: boolean; // supports image input
  json: boolean; // supports json output
  free: boolean; // rated free tier
}

export interface RatingsFile {
  schema: string; // ratings document schema
  ratings: RatingRow[]; // model rating entries
  best: { id: string; rating_milli: number }[]; // per-request ceiling
}

export interface Requirement {
  tools?: boolean; // requires function calling
  vision?: boolean; // requires image understanding
  json?: boolean; // requires structured json
  min_context?: number; // minimum token context
}

export interface WorkloadRequest {
  id: string; // workload request id
  api: 'openai' | 'anthropic'; // client wire dialect
  stream: boolean; // sse streaming flag
  weight: 1 | 2; // scoring weight multiplier
  requires: Requirement; // capability constraints
}

export interface WorkloadFile {
  schema: string; // workload document schema
  manifest: WorkloadRequest[]; // ordered workload items
}

export interface Brief {
  directory: ProviderDirectory; // parsed directory doc
  ratings: RatingsFile | null; // parsed ratings doc
  workload: WorkloadFile | null; // parsed workload doc
  error: string | null; // brief load diagnostic
}

export interface ProbedModel {
  id: string; // probed model slug
  display_name: string; // human readable name
  free: boolean | null; // probed free status
  context_length: number | null; // probed context limit
  tools: boolean | null; // probed tool support
  vision: boolean | null; // probed vision support
  json: boolean | null; // probed json support
}

export interface Probe {
  provider: ProviderDirectoryEntry; // target provider entry
  ok: boolean; // probe succeeded flag
  status: number; // http status code
  models: ProbedModel[]; // probed model list
  error: string | null; // probe error message
}

export interface Capabilities {
  context_length: number | null; // effective context window
  tools: boolean | null; // effective tool capability
  vision: boolean | null; // effective vision capability
  json: boolean | null; // effective json capability
}

export function joinBase(envOrigin: string, rawBase: string): string {
  const trimmedBase = rawBase.trim();
  if (/^https?:\/\//i.test(trimmedBase)) return trimmedBase.replace(/\/+$/, '');
  const cleanPrefix = envOrigin.replace(/\/+$/, '');
  if (trimmedBase === '') return cleanPrefix;
  return `${cleanPrefix}${trimmedBase.startsWith('/') ? '' : '/'}${trimmedBase.replace(/\/+$/, '')}`;
}

export function authHeaders(scheme: AuthScheme, secretKey: string): Record<string, string> {
  if (secretKey === '') return {}; // omit empty credentials
  if (scheme === 'bearer') return { authorization: `Bearer ${secretKey}` };
  if (scheme === 'x-api-key') return { 'x-api-key': secretKey };
  if (scheme === 'x-goog-api-key') return { 'x-goog-api-key': secretKey };
  return {}; // no auth header required
}

function extractDocs(rawMarkdown: string): Json[] {
  const docsFound: Json[] = [];
  for (let openIdx = rawMarkdown.indexOf('{'); openIdx !== -1; openIdx = rawMarkdown.indexOf('{', openIdx + 1)) {
    const closeIdx = findClosingBrace(rawMarkdown, openIdx);
    if (closeIdx === -1) continue; // unbalanced brace
    try {
      const parsedObj: unknown = JSON.parse(rawMarkdown.slice(openIdx, closeIdx + 1));
      if (isRecord(parsedObj) && typeof parsedObj.schema === 'string') docsFound.push(parsedObj);
    } catch {
      // ignore non-json blocks
    }
  }
  return docsFound;
}

function findClosingBrace(rawMarkdown: string, startOffset: number): number {
  let braceDepth = 0;
  let insideQuotes = false;
  for (let pos = startOffset; pos < rawMarkdown.length; pos++) {
    const glyph = rawMarkdown[pos];
    if (insideQuotes) {
      if (glyph === '\\') pos += 1;
      else if (glyph === '"') insideQuotes = false;
    } else if (glyph === '"') insideQuotes = true;
    else if (glyph === '{') braceDepth += 1;
    else if (glyph === '}') {
      braceDepth -= 1;
      if (braceDepth === 0) return pos;
    }
  }
  return -1; // no matching brace
}

const asStr = (val: unknown): string => (typeof val === 'string' ? val : '');
const asNum = (val: unknown): number | null => (typeof val === 'number' && Number.isFinite(val) ? val : null);
const asBool = (val: unknown): boolean | null => (typeof val === 'boolean' ? val : null);

function decodeProvider(item: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(item)) return null; // require json object
  const provId = asStr(item.id);
  const provWire = item.wire === 'openai' || item.wire === 'anthropic' || item.wire === 'gemini' ? item.wire : null;
  if (provId === '' || provWire === null) return null;
  const provAuth = item.auth === 'bearer' || item.auth === 'x-api-key' || item.auth === 'x-goog-api-key' || item.auth === 'none' ? item.auth : 'bearer';
  const provBase = asStr(item.path) || asStr(item.base_path) || asStr(item.base_url) || asStr(item.base);
  return { id: provId, name: asStr(item.name) || provId, wire: provWire, auth: provAuth, base: provBase, key_env: asStr(item.key_env), key: asStr(item.key) };
}

function decodeRating(item: unknown): RatingRow | null {
  if (!isRecord(item)) return null; // require rating object
  const modelSlug = asStr(item.model);
  const provSlug = asStr(item.provider);
  if (modelSlug === '' || provSlug === '') return null;
  return {
    model: modelSlug,
    provider: provSlug,
    rating_milli: typeof item.rating_milli === 'number' && Number.isFinite(item.rating_milli) ? item.rating_milli : 0,
    context_length: asNum(item.context_length),
    tools: item.tools === true,
    vision: item.vision === true,
    json: item.json === true,
    free: item.free === true,
  };
}

function decodeWorkloadItem(item: unknown): WorkloadRequest | null {
  if (!isRecord(item)) return null; // require workload object
  const reqId = asStr(item.id);
  if (reqId === '') return null;
  const reqCaps = isRecord(item.requires) ? item.requires : {};
  return {
    id: reqId,
    api: item.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: item.stream === true,
    weight: item.weight === 2 ? 2 : 1,
    requires: {
      tools: reqCaps.tools === true,
      vision: reqCaps.vision === true,
      json: reqCaps.json === true,
      ...(typeof reqCaps.min_context === 'number' && Number.isFinite(reqCaps.min_context) ? { min_context: reqCaps.min_context } : {}),
    },
  };
}

export function readBrief(briefPath: string): Brief {
  const emptyDir: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (briefPath === '') return { directory: emptyDir, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let rawContent: string;
  try {
    rawContent = readUtf8File(briefPath, 'utf8');
  } catch (err) {
    return { directory: emptyDir, ratings: null, workload: null, error: `cannot read ${briefPath}: ${(err as Error).message}` };
  }
  const allDocs = extractDocs(rawContent);
  const dirDoc = allDocs.find((doc) => doc.schema === PROVIDERS_SCHEMA && Array.isArray(doc.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { directory: emptyDir, ratings: null, workload: null, error: `the brief at ${briefPath} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const parsedProviders = dirDoc.providers.flatMap((entry) => {
    const decoded = decodeProvider(entry);
    return decoded === null ? [] : [decoded];
  });
  const rateDoc = allDocs.find((doc) => doc.schema === RATINGS_SCHEMA && Array.isArray(doc.ratings)) ?? null;
  const rawRatings = rateDoc !== null && Array.isArray(rateDoc.ratings) ? rateDoc.ratings : null;
  const parsedRatings: RatingsFile | null =
    rateDoc === null || rawRatings === null
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: rawRatings.flatMap((entry) => {
            const decoded = decodeRating(entry);
            return decoded === null ? [] : [decoded];
          }),
          best: Array.isArray(rateDoc.best)
            ? rateDoc.best.flatMap((entry) => {
                if (!isRecord(entry) || asStr(entry.id) === '') return [];
                return [{ id: asStr(entry.id), rating_milli: asNum(entry.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workDoc = allDocs.find((doc) => doc.schema === WORKLOAD_SCHEMA && Array.isArray(doc.manifest)) ?? null;
  const parsedWorkload: WorkloadFile | null =
    workDoc === null || !Array.isArray(workDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workDoc.manifest.flatMap((entry) => {
            const decoded = decodeWorkloadItem(entry);
            return decoded === null ? [] : [decoded];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers: parsedProviders }, ratings: parsedRatings, workload: parsedWorkload, error: null };
}

export function ratingOfModel(ratingsFile: RatingsFile | null, provId: string, modelId: string): RatingRow | null {
  if (ratingsFile === null) return null; // no ratings loaded
  return ratingsFile.ratings.find((row) => row.provider === provId && row.model === modelId) ?? null;
}

export function modelFree(probed: ProbedModel, rateRow: RatingRow | null): boolean {
  if (rateRow !== null) return rateRow.free; // prefer rating table free flag
  return probed.free ?? false;
}

export function capabilitiesOf(probed: ProbedModel, rateRow: RatingRow | null): Capabilities {
  return {
    context_length: rateRow?.context_length ?? probed.context_length ?? null,
    tools: rateRow?.tools ?? probed.tools ?? null,
    vision: rateRow?.vision ?? probed.vision ?? null,
    json: rateRow?.json ?? probed.json ?? null,
  };
}

export function meetsRequirements(req: Requirement, modelCaps: Capabilities): boolean {
  if (req.tools === true && modelCaps.tools !== true) return false;
  if (req.vision === true && modelCaps.vision !== true) return false;
  if (req.json === true && modelCaps.json !== true) return false;
  if (req.min_context !== undefined && (modelCaps.context_length === null || modelCaps.context_length < req.min_context)) return false;
  return true; // all constraints satisfied
}

function isZeroPricing(rawObj: Json): boolean | null {
  const priceMap = isRecord(rawObj.pricing) ? rawObj.pricing : null;
  if (priceMap === null) return null;
  const nums = Object.values(priceMap).flatMap((val) => {
    if (typeof val === 'number') return [val];
    if (typeof val === 'string' && val.trim() !== '') return [Number(val)];
    return [];
  });
  if (nums.length === 0) return null;
  return nums.every((num) => Number.isFinite(num) && num === 0);
}

function decodeProbedModel(wireKind: Wire, rawItem: unknown): ProbedModel | null {
  if (!isRecord(rawItem)) return null;
  const modelId = wireKind === 'gemini' ? asStr(rawItem.name).replace(/^models\//, '') : asStr(rawItem.id);
  if (modelId === '') return null;
  const ctxWindow = asNum(rawItem.context_length) ?? asNum(rawItem.context_window) ?? asNum(rawItem.inputTokenLimit);
  const label = asStr(rawItem.display_name) || asStr(rawItem.displayName) || modelId;
  return {
    id: modelId,
    display_name: label,
    free: asBool(rawItem.free) ?? isZeroPricing(rawItem),
    context_length: ctxWindow,
    tools: asBool(rawItem.tools),
    vision: asBool(rawItem.vision),
    json: asBool(rawItem.json),
  };
}

function extractModelList(wireKind: Wire, payloadBody: unknown): ProbedModel[] {
  if (!isRecord(payloadBody)) return [];
  const rawArr = wireKind === 'gemini' ? payloadBody.models : payloadBody.data;
  if (!Array.isArray(rawArr)) return [];
  return rawArr.flatMap((entry) => {
    const parsed = decodeProbedModel(wireKind, entry);
    return parsed === null ? [] : [parsed];
  });
}

export async function probeProvider(
  envOrigin: string,
  provEntry: ProviderDirectoryEntry,
  timeoutMs = 5000,
  ratingsFile: RatingsFile | null = null,
): Promise<Probe> {
  if (envOrigin.trim() === '' && ratingsFile !== null) {
    const fallbackModels: ProbedModel[] = ratingsFile.ratings
      .filter((row) => row.provider === provEntry.id)
      .map((row) => ({
        id: row.model,
        display_name: row.model,
        free: row.free,
        context_length: row.context_length,
        tools: row.tools,
        vision: row.vision,
        json: row.json,
      }));
    return { provider: provEntry, ok: true, status: 200, models: fallbackModels, error: null };
  }
  const baseEndpoint = joinBase(envOrigin, provEntry.base);
  const modelsUrl = provEntry.wire === 'gemini' ? `${baseEndpoint}/v1beta/models` : `${baseEndpoint}/v1/models`;
  const reqHeaders: Record<string, string> = { accept: 'application/json', ...authHeaders(provEntry.auth, provEntry.key) };
  if (provEntry.wire === 'anthropic') reqHeaders['anthropic-version'] = '2023-06-01';
  try {
    const httpRes = await fetch(modelsUrl, { headers: reqHeaders, signal: AbortSignal.timeout(timeoutMs) });
    if (!httpRes.ok) return { provider: provEntry, ok: false, status: httpRes.status, models: [], error: `GET ${modelsUrl} answered HTTP ${httpRes.status}` };
    const jsonPayload: unknown = await httpRes.json();
    return { provider: provEntry, ok: true, status: httpRes.status, models: extractModelList(provEntry.wire, jsonPayload), error: null };
  } catch (err) {
    return { provider: provEntry, ok: false, status: 0, models: [], error: `GET ${modelsUrl} failed: ${(err as Error).message}` };
  }
}

export function buildCatalog(_envOrigin: string, probeResults: readonly Probe[], ratingsFile: RatingsFile | null): CatalogReport {
  const catalogProviders: CatalogProvider[] = [];
  for (const probeItem of probeResults) {
    if (!probeItem.ok) continue; // skip failed probes
    const seenIds = new Set<string>();
    const catalogModels: CatalogModel[] = [];
    for (const mdl of [...probeItem.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seenIds.has(mdl.id)) continue;
      seenIds.add(mdl.id);
      const rateRow = ratingOfModel(ratingsFile, probeItem.provider.id, mdl.id);
      catalogModels.push({ id: mdl.id, free: modelFree(mdl, rateRow), context_length: rateRow?.context_length ?? mdl.context_length ?? null });
    }
    catalogProviders.push({ id: probeItem.provider.id, base_url: probeItem.provider.base, key_env: probeItem.provider.key_env, wire: probeItem.provider.wire, models: catalogModels });
  }
  return { providers: catalogProviders };
}
