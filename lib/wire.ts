import { isRecord, type Json, type Wire } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export interface NormalizedImage {
  kind: 'base64' | 'url';
  mediaType: string;
  data: string;
  url: string;
}

export interface NormalizedTool {
  name: string;
  description: string;
  schema: Json;
}

export interface NormalizedRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  systemText: string;
  userText: string;
  images: NormalizedImage[];
  tools: NormalizedTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

export type StopFamily = 'stop' | 'length' | 'tools' | 'filter';

export interface NormalizedAnswer {
  text: string;
  stopFamily: StopFamily;
  inputTokens: number;
  outputTokens: number;
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function parseDataUrl(url: string): NormalizedImage {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) {
    return { kind: 'base64', mediaType: m[1] || 'image/png', data: m[2], url };
  }
  return { kind: 'url', mediaType: 'image/png', data: '', url };
}

export function parseClientRequest(api: 'openai' | 'anthropic', body: Json): NormalizedRequest {
  if (api === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    let userText = '';
    const images: NormalizedImage[] = [];
    for (const msg of messages) {
      if (msg.role === 'system') {
        const text = blocksText(msg.content);
        if (text !== '') systemParts.push(text);
      } else if (msg.role === 'user') {
        userText = blocksText(msg.content);
        if (Array.isArray(msg.content)) {
          for (const block of msg.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string' && block.image_url.url !== '') {
              images.push(parseDataUrl(block.image_url.url));
            }
          }
        }
      }
    }
    const tools: NormalizedTool[] = [];
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
          tools.push({
            name: tool.function.name,
            description: typeof tool.function.description === 'string' ? tool.function.description : '',
            schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
          });
        }
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      api: 'openai',
      stream: body.stream === true,
      systemText: systemParts.join('\n'),
      userText,
      images,
      tools,
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      jsonMode: format === 'json_object' || format === 'json_schema',
      temperature: numberOrNull(body.temperature),
    };
  }

  // Anthropic client request
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  const images: NormalizedImage[] = [];
  for (const msg of messages) {
    if (msg.role !== 'user') continue;
    userText = blocksText(msg.content);
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          const mediaType = typeof src.media_type === 'string' && src.media_type !== '' ? src.media_type : 'image/png';
          images.push({ kind: 'base64', mediaType, data: src.data, url: `data:${mediaType};base64,${src.data}` });
        } else if (src.type === 'url' && typeof src.url === 'string' && src.url !== '') {
          images.push(parseDataUrl(src.url));
        }
      }
    }
  }
  const tools: NormalizedTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        tools.push({
          name: tool.name,
          description: typeof tool.description === 'string' ? tool.description : '',
          schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  return {
    api: 'anthropic',
    stream: body.stream === true,
    systemText,
    userText,
    images,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
  };
}

export function buildUpstreamBody(wire: Wire, model: string, req: NormalizedRequest): Json {
  if (wire === 'openai') {
    const messages: Json[] = [];
    if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
    if (req.images.length > 0) {
      const parts: Json[] = [{ type: 'text', text: req.userText }];
      for (const img of req.images) {
        const url = img.kind === 'base64' ? `data:${img.mediaType};base64,${img.data}` : img.url;
        parts.push({ type: 'image_url', image_url: { url } });
      }
      messages.push({ role: 'user', content: parts });
    } else {
      messages.push({ role: 'user', content: req.userText });
    }
    const out: Json = { model, messages };
    if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
    if (req.temperature !== null) out.temperature = req.temperature;
    if (req.stops.length > 0) out.stop = req.stops;
    if (req.tools.length > 0) {
      out.tools = req.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.schema },
      }));
    }
    if (req.jsonMode) out.response_format = { type: 'json_object' };
    return out;
  }

  if (wire === 'anthropic') {
    const content: unknown =
      req.images.length > 0
        ? [
            { type: 'text', text: req.userText },
            ...req.images.map((img) =>
              img.kind === 'base64'
                ? { type: 'image', source: { type: 'base64', media_type: img.mediaType, data: img.data } }
                : { type: 'image', source: { type: 'url', url: img.url } },
            ),
          ]
        : req.userText;
    const out: Json = {
      model,
      messages: [{ role: 'user', content }],
    };
    if (req.systemText !== '') out.system = req.systemText;
    if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
    if (req.temperature !== null) out.temperature = req.temperature;
    if (req.stops.length > 0) out.stop_sequences = req.stops;
    if (req.tools.length > 0) {
      out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
    }
    return out;
  }

  // Gemini upstream wire
  const parts: Json[] = [{ text: req.userText }];
  for (const img of req.images) {
    if (img.kind === 'base64') parts.push({ inlineData: { mimeType: img.mediaType, data: img.data } });
    else parts.push({ fileData: { fileUri: img.url } });
  }
  const out: Json = {
    contents: [{ role: 'user', parts }],
  };
  if (req.systemText !== '') {
    out.systemInstruction = { parts: [{ text: req.systemText }] };
  }
  if (req.tools.length > 0) {
    out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const gc: Json = {};
  if (req.maxTokens !== null) gc.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) gc.temperature = req.temperature;
  if (req.stops.length > 0) gc.stopSequences = req.stops;
  if (req.jsonMode) gc.responseMimeType = 'application/json';
  if (Object.keys(gc).length > 0) out.generationConfig = gc;
  return out;
}

function stopFamilyFrom(reason: unknown): StopFamily {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tools';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 'filter';
  return 'stop';
}

export function parseUpstreamAnswer(wire: Wire, body: Json): NormalizedAnswer {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const msg = isRecord(choice.message) ? choice.message : {};
    const text = typeof msg.content === 'string' ? msg.content : blocksText(msg.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      stopFamily: stopFamilyFrom(choice.finish_reason),
      inputTokens: numberOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numberOrNull(usage.completion_tokens) ?? 0,
    };
  }

  if (wire === 'anthropic') {
    const text = blocksText(body.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      stopFamily: stopFamilyFrom(body.stop_reason),
      inputTokens: numberOrNull(usage.input_tokens) ?? 0,
      outputTokens: numberOrNull(usage.output_tokens) ?? 0,
    };
  }

  // Gemini upstream response
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join('\n'),
    stopFamily: stopFamilyFrom(cand.finishReason),
    inputTokens: numberOrNull(meta.promptTokenCount) ?? 0,
    outputTokens: numberOrNull(meta.candidatesTokenCount) ?? 0,
  };
}

const OPENAI_STOP: Record<StopFamily, string> = {
  stop: 'stop',
  length: 'length',
  tools: 'tool_calls',
  filter: 'content_filter',
};

const ANTHROPIC_STOP: Record<StopFamily, string> = {
  stop: 'end_turn',
  length: 'max_tokens',
  tools: 'tool_use',
  filter: 'refusal',
};

export function emitClientJson(api: 'openai' | 'anthropic', ans: NormalizedAnswer, model: string): Json {
  if (api === 'openai') {
    return {
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: ans.text }, finish_reason: OPENAI_STOP[ans.stopFamily], logprobs: null }],
      usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
    };
  }
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: ANTHROPIC_STOP[ans.stopFamily],
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export function emitClientSse(api: 'openai' | 'anthropic', ans: NormalizedAnswer, model: string): string {
  if (api === 'openai') {
    const chunk = (delta: Json, finish: string | null): Json => ({
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const frames = [
      sseFrame(null, chunk({ role: 'assistant', content: '' }, null)),
      sseFrame(null, chunk({ content: ans.text }, null)),
      sseFrame(null, chunk({}, OPENAI_STOP[ans.stopFamily])),
      sseFrame(null, {
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
    return frames.join('');
  }

  const frames = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ans.text } }),
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTHROPIC_STOP[ans.stopFamily], stop_sequence: null },
      usage: { output_tokens: ans.outputTokens },
    }),
    sseFrame('message_stop', { type: 'message_stop' }),
  ];
  return frames.join('');
}
