import { isRecord, type Json, type Wire } from './providers.ts'; // wire protocol helpers

export interface ServedModel {
  id: string; // served model identifier
  provider: string; // owning provider id
  display_name: string; // model display title
}

const CREATED_EPOCH_SEC = 1767225600; // deterministic openai timestamp
const CREATED_ISO_STR = '2026-01-01T00:00:00Z'; // deterministic anthropic timestamp

export function openAIModelList(servedList: readonly ServedModel[]): Json {
  return {
    object: 'list', // openai list envelope
    data: servedList.map((item) => ({ id: item.id, object: 'model', created: CREATED_EPOCH_SEC, owned_by: item.provider })),
  };
}

export function anthropicModelList(servedList: readonly ServedModel[]): Json {
  return {
    data: servedList.map((item) => ({ type: 'model', id: item.id, display_name: item.display_name, created_at: CREATED_ISO_STR })),
    has_more: false, // complete single-page list
  };
}

export function anthropicError(errKind: string, errMsg: string): Json {
  return { type: 'error', error: { type: errKind, message: errMsg } }; // anthropic error shape
}

export function openAIError(errMsg: string, errKind: string, errCode: string | null): Json {
  return { error: { message: errMsg, type: errKind, code: errCode } }; // openai error shape
}

export interface NormalizedImage {
  kind: 'base64' | 'url'; // image encoding mode
  mediaType: string; // mime media type
  data: string; // base64 payload bytes
  url: string; // source image url
}

export interface NormalizedTool {
  name: string; // function tool name
  description: string; // tool description text
  schema: Json; // json schema parameters
}

export interface NormalizedRequest {
  api: 'openai' | 'anthropic'; // caller api dialect
  stream: boolean; // caller requested sse
  systemText: string; // combined system prompt
  userText: string; // combined user prompt
  images: NormalizedImage[]; // extracted vision inputs
  tools: NormalizedTool[]; // extracted tool declarations
  maxTokens: number | null; // output token cap
  stops: string[]; // custom stop strings
  jsonMode: boolean; // structured json mode
  temperature: number | null; // sampling temperature
}

export type StopFamily = 'stop' | 'length' | 'tools' | 'filter'; // canonical finish category

export interface NormalizedAnswer {
  text: string; // assistant text reply
  stopFamily: StopFamily; // normalized finish reason
  inputTokens: number; // prompt token count
  outputTokens: number; // completion token count
}

function extractTextBlocks(rawContent: unknown): string {
  if (typeof rawContent === 'string') return rawContent;
  if (!Array.isArray(rawContent)) return '';
  const textFragments: string[] = [];
  for (const part of rawContent) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') textFragments.push(part.text);
  }
  return textFragments.join('\n');
}

const finiteNum = (val: unknown): number | null => (typeof val === 'number' && Number.isFinite(val) ? val : null);

function toStrArray(val: unknown): string[] {
  if (typeof val === 'string') return [val];
  if (!Array.isArray(val)) return [];
  return val.filter((entry): entry is string => typeof entry === 'string');
}

function decodeImageUri(rawUri: string): NormalizedImage {
  const matchData = /^data:([^;,]*);base64,(.*)$/s.exec(rawUri);
  if (matchData !== null) {
    return { kind: 'base64', mediaType: matchData[1] || 'image/png', data: matchData[2], url: rawUri };
  }
  return { kind: 'url', mediaType: 'image/png', data: '', url: rawUri };
}

export function parseClientRequest(clientApi: 'openai' | 'anthropic', reqBody: Json): NormalizedRequest {
  if (clientApi === 'openai') {
    const msgList = Array.isArray(reqBody.messages) ? reqBody.messages.filter(isRecord) : [];
    const sysSegments: string[] = [];
    let usrPrompt = '';
    const imgAttachments: NormalizedImage[] = [];
    for (const msgItem of msgList) {
      if (msgItem.role === 'system') {
        const sysTxt = extractTextBlocks(msgItem.content);
        if (sysTxt !== '') sysSegments.push(sysTxt);
      } else if (msgItem.role === 'user') {
        usrPrompt = extractTextBlocks(msgItem.content);
        if (Array.isArray(msgItem.content)) {
          for (const part of msgItem.content) {
            if (!isRecord(part)) continue;
            if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string' && part.image_url.url !== '') {
              imgAttachments.push(decodeImageUri(part.image_url.url));
            }
          }
        }
      }
    }
    const toolSpecs: NormalizedTool[] = [];
    if (Array.isArray(reqBody.tools)) {
      for (const rawTool of reqBody.tools) {
        if (isRecord(rawTool) && isRecord(rawTool.function) && typeof rawTool.function.name === 'string') {
          toolSpecs.push({
            name: rawTool.function.name,
            description: typeof rawTool.function.description === 'string' ? rawTool.function.description : '',
            schema: isRecord(rawTool.function.parameters) ? rawTool.function.parameters : { type: 'object', properties: {} },
          });
        }
      }
    }
    const respFmt = isRecord(reqBody.response_format) ? reqBody.response_format.type : undefined;
    return {
      api: 'openai',
      stream: reqBody.stream === true,
      systemText: sysSegments.join('\n'),
      userText: usrPrompt,
      images: imgAttachments,
      tools: toolSpecs,
      maxTokens: finiteNum(reqBody.max_tokens) ?? finiteNum(reqBody.max_completion_tokens),
      stops: toStrArray(reqBody.stop),
      jsonMode: respFmt === 'json_object' || respFmt === 'json_schema',
      temperature: finiteNum(reqBody.temperature),
    };
  }

  const anthSystem = extractTextBlocks(reqBody.system);
  const anthMsgs = Array.isArray(reqBody.messages) ? reqBody.messages.filter(isRecord) : [];
  let anthUser = '';
  const anthImages: NormalizedImage[] = [];
  for (const msgItem of anthMsgs) {
    if (msgItem.role !== 'user') continue;
    anthUser = extractTextBlocks(msgItem.content);
    if (Array.isArray(msgItem.content)) {
      for (const part of msgItem.content) {
        if (!isRecord(part) || part.type !== 'image' || !isRecord(part.source)) continue;
        const imgSrc = part.source;
        if (imgSrc.type === 'base64' && typeof imgSrc.data === 'string') {
          const mimeStr = typeof imgSrc.media_type === 'string' && imgSrc.media_type !== '' ? imgSrc.media_type : 'image/png';
          anthImages.push({ kind: 'base64', mediaType: mimeStr, data: imgSrc.data, url: `data:${mimeStr};base64,${imgSrc.data}` });
        } else if (imgSrc.type === 'url' && typeof imgSrc.url === 'string' && imgSrc.url !== '') {
          anthImages.push(decodeImageUri(imgSrc.url));
        }
      }
    }
  }
  const anthTools: NormalizedTool[] = [];
  if (Array.isArray(reqBody.tools)) {
    for (const rawTool of reqBody.tools) {
      if (isRecord(rawTool) && typeof rawTool.name === 'string') {
        anthTools.push({
          name: rawTool.name,
          description: typeof rawTool.description === 'string' ? rawTool.description : '',
          schema: isRecord(rawTool.input_schema) ? rawTool.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  return {
    api: 'anthropic',
    stream: reqBody.stream === true,
    systemText: anthSystem,
    userText: anthUser,
    images: anthImages,
    tools: anthTools,
    maxTokens: finiteNum(reqBody.max_tokens),
    stops: toStrArray(reqBody.stop_sequences),
    jsonMode: false,
    temperature: finiteNum(reqBody.temperature),
  };
}

export function buildUpstreamBody(targetWire: Wire, modelSlug: string, normReq: NormalizedRequest): Json {
  if (targetWire === 'openai') {
    const oaiMsgs: Json[] = [];
    if (normReq.systemText !== '') oaiMsgs.push({ role: 'system', content: normReq.systemText });
    if (normReq.images.length > 0) {
      const contentParts: Json[] = [{ type: 'text', text: normReq.userText }];
      for (const imgItem of normReq.images) {
        const imgUri = imgItem.kind === 'base64' ? `data:${imgItem.mediaType};base64,${imgItem.data}` : imgItem.url;
        contentParts.push({ type: 'image_url', image_url: { url: imgUri } });
      }
      oaiMsgs.push({ role: 'user', content: contentParts });
    } else {
      oaiMsgs.push({ role: 'user', content: normReq.userText });
    }
    const oaiPayload: Json = { model: modelSlug, messages: oaiMsgs };
    if (normReq.maxTokens !== null) oaiPayload.max_tokens = normReq.maxTokens;
    if (normReq.temperature !== null) oaiPayload.temperature = normReq.temperature;
    if (normReq.stops.length > 0) oaiPayload.stop = normReq.stops;
    if (normReq.tools.length > 0) {
      oaiPayload.tools = normReq.tools.map((toolItem) => ({
        type: 'function',
        function: { name: toolItem.name, description: toolItem.description, parameters: toolItem.schema },
      }));
    }
    if (normReq.jsonMode) oaiPayload.response_format = { type: 'json_object' };
    return oaiPayload;
  }

  if (targetWire === 'anthropic') {
    const userContent: unknown =
      normReq.images.length > 0
        ? [
            { type: 'text', text: normReq.userText },
            ...normReq.images.map((imgItem) =>
              imgItem.kind === 'base64'
                ? { type: 'image', source: { type: 'base64', media_type: imgItem.mediaType, data: imgItem.data } }
                : { type: 'image', source: { type: 'url', url: imgItem.url } },
            ),
          ]
        : normReq.userText;
    const anthPayload: Json = {
      model: modelSlug,
      messages: [{ role: 'user', content: userContent }],
    };
    if (normReq.systemText !== '') anthPayload.system = normReq.systemText;
    if (normReq.maxTokens !== null) anthPayload.max_tokens = normReq.maxTokens;
    if (normReq.temperature !== null) anthPayload.temperature = normReq.temperature;
    if (normReq.stops.length > 0) anthPayload.stop_sequences = normReq.stops;
    if (normReq.tools.length > 0) {
      anthPayload.tools = normReq.tools.map((toolItem) => ({ name: toolItem.name, description: toolItem.description, input_schema: toolItem.schema }));
    }
    return anthPayload;
  }

  const gemParts: Json[] = [{ text: normReq.userText }];
  for (const imgItem of normReq.images) {
    if (imgItem.kind === 'base64') gemParts.push({ inlineData: { mimeType: imgItem.mediaType, data: imgItem.data } });
    else gemParts.push({ fileData: { fileUri: imgItem.url } });
  }
  const gemPayload: Json = {
    contents: [{ role: 'user', parts: gemParts }],
  };
  if (normReq.systemText !== '') {
    gemPayload.systemInstruction = { parts: [{ text: normReq.systemText }] };
  }
  if (normReq.tools.length > 0) {
    gemPayload.tools = [{ functionDeclarations: normReq.tools.map((toolItem) => ({ name: toolItem.name, description: toolItem.description, parameters: toolItem.schema })) }];
  }
  const genCfg: Json = {};
  if (normReq.maxTokens !== null) genCfg.maxOutputTokens = normReq.maxTokens;
  if (normReq.temperature !== null) genCfg.temperature = normReq.temperature;
  if (normReq.stops.length > 0) genCfg.stopSequences = normReq.stops;
  if (normReq.jsonMode) genCfg.responseMimeType = 'application/json';
  if (Object.keys(genCfg).length > 0) gemPayload.generationConfig = genCfg;
  return gemPayload;
}

function classifyStopReason(rawReason: unknown): StopFamily {
  if (rawReason === 'length' || rawReason === 'max_tokens' || rawReason === 'MAX_TOKENS') return 'length';
  if (rawReason === 'tool_calls' || rawReason === 'function_call' || rawReason === 'tool_use') return 'tools';
  if (rawReason === 'content_filter' || rawReason === 'refusal' || rawReason === 'SAFETY') return 'filter';
  return 'stop'; // default normal completion
}

export function parseUpstreamAnswer(sourceWire: Wire, respBody: Json): NormalizedAnswer {
  if (sourceWire === 'openai') {
    const choiceArr = Array.isArray(respBody.choices) ? respBody.choices : [];
    const firstChoice = isRecord(choiceArr[0]) ? choiceArr[0] : {};
    const msgObj = isRecord(firstChoice.message) ? firstChoice.message : {};
    const answerTxt = typeof msgObj.content === 'string' ? msgObj.content : extractTextBlocks(msgObj.content);
    const usageObj = isRecord(respBody.usage) ? respBody.usage : {};
    return {
      text: answerTxt,
      stopFamily: classifyStopReason(firstChoice.finish_reason),
      inputTokens: finiteNum(usageObj.prompt_tokens) ?? 0,
      outputTokens: finiteNum(usageObj.completion_tokens) ?? 0,
    };
  }

  if (sourceWire === 'anthropic') {
    const answerTxt = extractTextBlocks(respBody.content);
    const usageObj = isRecord(respBody.usage) ? respBody.usage : {};
    return {
      text: answerTxt,
      stopFamily: classifyStopReason(respBody.stop_reason),
      inputTokens: finiteNum(usageObj.input_tokens) ?? 0,
      outputTokens: finiteNum(usageObj.output_tokens) ?? 0,
    };
  }

  const candArr = Array.isArray(respBody.candidates) ? respBody.candidates : [];
  const firstCand = isRecord(candArr[0]) ? candArr[0] : {};
  const candContent = isRecord(firstCand.content) ? firstCand.content : {};
  const candParts = Array.isArray(candContent.parts) ? candContent.parts : [];
  const textChunks: string[] = [];
  for (const partItem of candParts) {
    if (isRecord(partItem) && typeof partItem.text === 'string') textChunks.push(partItem.text);
  }
  const gemMeta = isRecord(respBody.usageMetadata) ? respBody.usageMetadata : {};
  return {
    text: textChunks.join('\n'),
    stopFamily: classifyStopReason(firstCand.finishReason),
    inputTokens: finiteNum(gemMeta.promptTokenCount) ?? 0,
    outputTokens: finiteNum(gemMeta.candidatesTokenCount) ?? 0,
  };
}

const OAI_FINISH_MAP: Record<StopFamily, string> = {
  stop: 'stop', // oai stop reason
  length: 'length', // oai length reason
  tools: 'tool_calls', // oai tool reason
  filter: 'content_filter', // oai filter reason
};

const ANTH_FINISH_MAP: Record<StopFamily, string> = {
  stop: 'end_turn', // anthropic end_turn
  length: 'max_tokens', // anthropic max_tokens
  tools: 'tool_use', // anthropic tool_use
  filter: 'refusal', // anthropic refusal
};

export function emitClientJson(clientApi: 'openai' | 'anthropic', normAns: NormalizedAnswer, modelSlug: string): Json {
  if (clientApi === 'openai') {
    return {
      id: 'chatcmpl-agg', // deterministic completion id
      object: 'chat.completion', // openai completion object
      created: CREATED_EPOCH_SEC,
      model: modelSlug,
      choices: [{ index: 0, message: { role: 'assistant', content: normAns.text }, finish_reason: OAI_FINISH_MAP[normAns.stopFamily], logprobs: null }],
      usage: { prompt_tokens: normAns.inputTokens, completion_tokens: normAns.outputTokens, total_tokens: normAns.inputTokens + normAns.outputTokens },
    };
  }
  return {
    id: 'msg_agg', // deterministic message id
    type: 'message', // anthropic message object
    role: 'assistant', // assistant role
    model: modelSlug,
    content: [{ type: 'text', text: normAns.text }],
    stop_reason: ANTH_FINISH_MAP[normAns.stopFamily],
    stop_sequence: null,
    usage: { input_tokens: normAns.inputTokens, output_tokens: normAns.outputTokens },
  };
}

function formatSseEvent(eventName: string | null, payloadData: unknown): string {
  return `${eventName === null ? '' : `event: ${eventName}\n`}data: ${JSON.stringify(payloadData)}\n\n`;
}

export function emitClientSse(clientApi: 'openai' | 'anthropic', normAns: NormalizedAnswer, modelSlug: string): string {
  if (clientApi === 'openai') {
    const buildDeltaChunk = (deltaObj: Json, finishReason: string | null): Json => ({
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: CREATED_EPOCH_SEC,
      model: modelSlug,
      choices: [{ index: 0, delta: deltaObj, finish_reason: finishReason }],
    });
    const oaiFrames = [
      formatSseEvent(null, buildDeltaChunk({ role: 'assistant', content: '' }, null)),
      formatSseEvent(null, buildDeltaChunk({ content: normAns.text }, null)),
      formatSseEvent(null, buildDeltaChunk({}, OAI_FINISH_MAP[normAns.stopFamily])),
      formatSseEvent(null, {
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: CREATED_EPOCH_SEC,
        model: modelSlug,
        choices: [],
        usage: { prompt_tokens: normAns.inputTokens, completion_tokens: normAns.outputTokens, total_tokens: normAns.inputTokens + normAns.outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
    return oaiFrames.join('');
  }

  const anthFrames = [
    formatSseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model: modelSlug,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: normAns.inputTokens, output_tokens: 0 },
      },
    }),
    formatSseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    formatSseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: normAns.text } }),
    formatSseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    formatSseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTH_FINISH_MAP[normAns.stopFamily], stop_sequence: null },
      usage: { output_tokens: normAns.outputTokens },
    }),
    formatSseEvent('message_stop', { type: 'message_stop' }),
  ];
  return anthFrames.join('');
}
