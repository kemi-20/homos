import { homoQuotes } from "./data/homo-quotes";

type AiBinding = {
	run: (
		model: string,
		input: {
			messages: { role: "system" | "user"; content: string }[];
			max_tokens?: number;
			temperature?: number;
		},
	) => Promise<unknown>;
};

type AssetsBinding = {
	fetch: (request: Request) => Promise<Response>;
};

type KvBinding = {
	get: (key: string, options?: { type?: "json" | "text" }) => Promise<unknown>;
	put: (key: string, value: string) => Promise<void>;
};

type WorkerContext = {
	waitUntil: (promise: Promise<unknown>) => void;
};

type Env = {
	AI: AiBinding;
	ASSETS: AssetsBinding;
	QUOTE_KV: KvBinding;
};

type DailyQuote = {
	text: string;
	source: string;
	generated: boolean;
	model: string;
	date: string;
	id?: string;
};

const MODEL = "@cf/ibm-granite/granite-4.0-h-micro";
const DAILY_QUOTE_COUNT = 10;
const HISTORY_QUOTE_COUNT = 100;

function getUtcDateKey(date = new Date()) {
	return date.toISOString().slice(0, 10);
}

function dayIndex(date = new Date()) {
	const start = Date.UTC(date.getUTCFullYear(), 0, 0);
	const current = Date.UTC(
		date.getUTCFullYear(),
		date.getUTCMonth(),
		date.getUTCDate(),
	);
	return Math.floor((current - start) / 86400000);
}

function fallbackQuote(date = new Date()): DailyQuote {
	const quote = homoQuotes[dayIndex(date) % homoQuotes.length] ?? homoQuotes[0];
	return {
		...quote,
		generated: false,
		model: "local-fallback",
		date: getUtcDateKey(date),
	};
}

function localQuotePool(date = new Date()): DailyQuote[] {
	const dateKey = getUtcDateKey(date);
	return homoQuotes.map((quote, index) => ({
		...quote,
		id: `local-${index}`,
		generated: false,
		model: "local-preset",
		date: dateKey,
	}));
}

function extractText(result: unknown) {
	if (typeof result === "string") return result;
	if (!result || typeof result !== "object") return "";

	const record = result as Record<string, unknown>;
	if (typeof record.response === "string") return record.response;
	if (typeof record.result === "string") return record.result;
	if (typeof record.text === "string") return record.text;

	const choices = record.choices;
	if (Array.isArray(choices)) {
		for (const choice of choices) {
			if (!choice || typeof choice !== "object") continue;
			const choiceRecord = choice as Record<string, unknown>;
			const message = choiceRecord.message;
			if (message && typeof message === "object") {
				const content = (message as Record<string, unknown>).content;
				if (typeof content === "string") return content;
			}
			if (typeof choiceRecord.text === "string") return choiceRecord.text;
		}
	}

	return "";
}

function cleanQuote(text: string) {
	return text
		.replace(/^["'“”‘’「」『』]+|["'“”‘’「」『』]+$/g, "")
		.replace(/\s+/g, " ")
		.trim();
}

function parseGeneratedLines(text: string) {
	return text
		.split(/\r?\n/)
		.map((line) =>
			cleanQuote(line.replace(/^\s*(?:[-*]|\d+[.)、：:]?)\s*/, "")),
		)
		.filter((line) => line.length >= 6 && line.length <= 80);
}

function uniqueByText(quotes: DailyQuote[]) {
	const seen = new Set<string>();
	const result: DailyQuote[] = [];
	for (const quote of quotes) {
		const key = quote.text.trim();
		if (!key || seen.has(key)) continue;
		seen.add(key);
		result.push(quote);
	}
	return result;
}

function pickRandom<T>(items: T[]) {
	if (!items.length) return undefined;
	const array = new Uint32Array(1);
	crypto.getRandomValues(array);
	return items[array[0] % items.length];
}

async function getJson<T>(kv: KvBinding, key: string, fallback: T): Promise<T> {
	const value = await kv.get(key, { type: "json" });
	return value == null ? fallback : (value as T);
}

async function saveQuotes(kv: KvBinding, key: string, quotes: DailyQuote[]) {
	await kv.put(key, JSON.stringify(quotes));
}

async function generateDailyQuotes(env: Env, date = new Date()) {
	const dateKey = getUtcDateKey(date);
	const result = await env.AI.run(MODEL, {
		temperature: 0.85,
		max_tokens: 520,
		messages: [
			{
				role: "system",
				content: "你是 HOMOS 网站的每日一言生成器。严格按要求输出，不解释。",
			},
			{
				role: "user",
				content:
					"生成 10 句中文互联网 homo / INM 风格的每日一言。要求：每句 20 到 40 个字；可以包含 114514、810、迫真、要素过多、Q.E.D. 等语境；不要露骨色情内容；不要仇恨、骚扰或人身攻击；每行一句；不要编号；不要标题。",
			},
		],
	});

	const generated = uniqueByText(
		parseGeneratedLines(extractText(result)).map((text, index) => ({
			text,
			source: "Workers AI / Granite Micro",
			generated: true,
			model: MODEL,
			date: dateKey,
			id: `ai-${dateKey}-${index}`,
		})),
	);

	const fallbackPool = localQuotePool(date);
	while (generated.length < DAILY_QUOTE_COUNT) {
		const fallback =
			fallbackPool[(dayIndex(date) + generated.length) % fallbackPool.length];
		generated.push({
			...fallback,
			id: `fallback-${dateKey}-${generated.length}`,
		});
	}

	return generated.slice(0, DAILY_QUOTE_COUNT);
}

async function ensureTodayQuotes(env: Env, date = new Date()) {
	const dateKey = getUtcDateKey(date);
	const dailyKey = `daily:${dateKey}`;
	const existing = await getJson<DailyQuote[]>(env.QUOTE_KV, dailyKey, []);
	if (existing.length >= DAILY_QUOTE_COUNT) {
		return existing.slice(0, DAILY_QUOTE_COUNT);
	}

	const todayQuotes = await generateDailyQuotes(env, date);
	await saveQuotes(env.QUOTE_KV, dailyKey, todayQuotes);

	const history = await getJson<DailyQuote[]>(env.QUOTE_KV, "history", []);
	const nextHistory = uniqueByText([...todayQuotes, ...history]).slice(
		0,
		HISTORY_QUOTE_COUNT,
	);
	await saveQuotes(env.QUOTE_KV, "history", nextHistory);
	return todayQuotes;
}

function jsonResponse(body: unknown, init: ResponseInit = {}) {
	const headers = new Headers(init.headers);
	headers.set("content-type", "application/json; charset=utf-8");
	headers.set("access-control-allow-origin", "*");
	headers.set("cache-control", "no-store");
	return new Response(JSON.stringify(body), { ...init, headers });
}

async function handleDailyQuote(request: Request, env: Env) {
	const url = new URL(request.url);
	const mode = url.searchParams.get("mode") === "reroll" ? "reroll" : "today";

	try {
		const todayQuotes = await ensureTodayQuotes(env);
		const history = await getJson<DailyQuote[]>(env.QUOTE_KV, "history", []);
		const pool =
			mode === "reroll"
				? uniqueByText([...history, ...localQuotePool()])
				: todayQuotes;

		return jsonResponse({
			mode,
			quote: pickRandom(pool) ?? fallbackQuote(),
			todayCount: todayQuotes.length,
			historyCount: history.length,
			presetCount: homoQuotes.length,
			poolSize: pool.length,
		});
	} catch (error) {
		const pool = localQuotePool();
		return jsonResponse(
			{
				mode,
				quote: pickRandom(pool) ?? fallbackQuote(),
				todayCount: 0,
				historyCount: 0,
				presetCount: homoQuotes.length,
				poolSize: pool.length,
				error: error instanceof Error ? error.message : "daily quote unavailable",
			},
			{ status: 200 },
		);
	}
}

export default {
	async fetch(request: Request, env: Env, _ctx: WorkerContext): Promise<Response> {
		const url = new URL(request.url);
		if (url.pathname === "/api/daily-quote") {
			return handleDailyQuote(request, env);
		}

		return env.ASSETS.fetch(request);
	},
};
