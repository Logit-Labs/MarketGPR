"""Deterministic GPR-role classifier — the Phase 2 lexicon core.

A dictionary method in the tradition of Caldara–Iacoviello's GPR index:
curated term lists plus explicit co-occurrence rules plus documented
precedence. The same DB plus the same lexicon version yields identical
labels — no LLM, no randomness, no wall clock in the output.

`decide(title, category, tags_json, name_distinct)` is the whole interface:
a pure function returning `(label, confidence, rationale)` for one series.
Rules fire in strict order; the first hit wins (see the docstring of
`decide`). `needs_review` is a query-time threshold decision made by the
caller, not a decision made here.

The four model labels are roles, not degrees of relatedness — see
Phase2.series_tables for the taxonomy. `background` is resolved
to `control`/`drop` downstream by traded volume (a fact in the database,
not a judgement about the subject).
"""

import json
import re
import unicodedata

LEXICON_VERSION = "v5"
RULES_MODEL = "rules-v5"          # model name written to series_labels/series_label_runs

# Match sources: title (weight 2), tags + category (weight 1).
_TITLE_WEIGHT = 2
_TAG_WEIGHT = 1

# EXCLUDE: any hit (title or tags or category) => background (0.85), overrides ALL other rules.
EXCLUDE = {
    # sports — league/sport names and strong sport phrases only; no ambiguous single words
    "nba","nfl","mlb","nhl","wnba","ufc","boxing","tennis","atp","wta","pga","lpga",
    "golf","soccer","football","basketball","baseball","hockey","cricket","f1","nascar",
    "formula 1","grand prix","esports","cs2","counter strike","dota","league of legends",
    "valorant","chess","olympics","olympic games","world cup","tournament","championship",
    "goalscorer","finisher","mvp","world series","super bowl","premier league","la liga",
    "serie a","bundesliga","fa cup","uefa","fifa","ncaa","match winner","set winner",
    "player of the match","group stage","round of 16","semifinal","draft pick","head coach",
    "cycling","tour de france","stage winner","triathlon","rugby","nrl","aussie rules",
    "volleyball","handball","water polo","swimming","skiing","snowboarding","figure skating",
    "speed skating","gymnastics","marathon","track and field","rowing","darts","snooker",
    # equity-index / price-chart bets
    "nasdaq","s&p 500","dow jones","dji","nky","nikkei","hang seng","dax","ftse",
    "cac 40","stock index","equity index","stock market",
    # sports-betting vocabulary
    "betting","odds","moneyline","parlay","point spread","prop bet","over under",
    "team total","player props",
    # entertainment awards/events
    "award show","red carpet","met gala","coachella","glastonbury",
    # crypto
    "bitcoin","btc","ethereum","eth","solana","sol","xrp","ripple","dogecoin","doge",
    "bnb","hype","shiba","meme coin","crypto","cryptocurrency","nft","defi","stablecoin",
    "usdt","tether",
    # entertainment / celebrity
    "oscar","grammy","emmy","billboard","box office","rotten tomatoes","netflix","disney",
    "marvel","spotify","tiktok","instagram","album","concert","festival","movie","film",
    "tv show","celebrity","influencer","streams","debut","love island","box office gross",
    # weather / climate — clear unambiguous terms only ("flood"/"storm" deliberately absent:
    # ambiguous with border/migration contexts; those fall to default-background (0.55) review)
    "rain","hurricane","tornado","earthquake","temperature","precipitation","snowfall",
    "heat wave","drought","snow","wind speed","rainfall","humidity","weather",
    # games / trivia
    "video game","videogame","minecraft","fortnite","wordle","trivia","gaming",
    # v4 — QA-verified (LLM QA on both exchanges): soccer fixtures by club
    # suffix, reality TV / song contests, AI-model releases, influencer
    # word-counts, health statistics. Bare "ai" is deliberately absent: an
    # "AI export ban" series is genuine geopolitics.
    "fc","afc","eurovision","agt","american idol","miss universe",
    "miss france","love is blind","beast games","america's got talent",
    "grok","claude","gemini","mythos","chatgpt","llm","artificial intelligence",
    "oscars","grammys","emmys","mr beast","mrbeast","snl","tnf","covid",
    "coronavirus",
    # v5 — QA-flagged sports/travel/tech specifics, restored narrow: the bare
    # words "sports"/"entertainment" are deliberately NOT excluded — a
    # "transgender sports ban" ballot-measure series is politics, and the
    # title match would silently kill it before the QA could see it.
    "lacrosse","mma","fight night","tsa","app ranking",
}

# ACTORS: countries/regions/leaders. Never standalone — fire only via co-occurrence rules.
# v5: US figures added — US domestic politics is geopolitical per the project
# definition (politics incl. domestic), so "Trump transgender sports ban?" is a
# policy series, not a sports one. Figures only, never the bare country word.
ACTORS = {
    "ukraine","russia","china","taiwan","iran","israel","palestine","gaza","hamas",
    "hezbollah","lebanon","syria","iraq","afghanistan","north korea","south korea",
    "venezuela","belarus","cuba","pakistan","india","turkey","armenia","azerbaijan",
    "georgia","moldova","serbia","kosovo","bosnia","britain","france","germany","poland",
    "japan","australia","brazil","mexico","nigeria","egypt","saudi arabia","qatar",
    "yemen","sudan","libya","ethiopia","myanmar","vietnam","philippines","indonesia",
    "kazakhstan","middle east","taiwan strait","south china sea","united kingdom","uk",
    "europe","eu","european union","nato",
    "putin","zelensky","xi jinping","kim jong un","netanyahu","khamenei","maduro",
    "lukashenko","macron","scholz","starmer","modi",
    "trump","biden","harris","vance","obama","pelosi","schumer","mcconnell",
}

# STRONG_ACTIONS: inherently geopolitical alone; no actor required.
STRONG_ACTIONS = {
    "war","wars","world war","cold war","nuclear war","ceasefire","cease fire","truce",
    "invasion","airstrike","air strike","missile","missiles","nuclear","nuke","sanction",
    "sanctions","embargo","nato","coup","assassination","occupation","annexation",
    "mobilization","genocide","war crimes","martial law","military operation","peace treaty",
    "peace talks","peace deal","prisoner exchange","prisoner swap","military aid","arms deal",
    "defense pact","security pact","geopolitics","geopolitical","geopolitical risk",
    "geopolitical conflict","military conflict","armed conflict","border dispute",
    "territorial dispute","chemical weapons","chemical attack","uprising","insurgency",
    "rebellion","military strike","military action","drone strike","rocket attack",
    "terrorist","terrorism","terrorist attack","regime change","proxy war","cyberattack",
    "war of aggression",
}

# ACTIONS: need an ACTOR in the same text (title only counts; tags-only action is ignored).
ACTIONS = {
    "attack","attacks","bombing","bomb","troops","soldiers","casualties","offensive",
    "occupied","escalation","conflict","refugee","refugees","evacuation","evacuations",
    "weapons","arms","rebel","rebels","militant","militants","deploy","deployed","military",
    "defense","tariff","tariffs","trade war","trade deal","import ban","export ban",
    "export controls","chip ban","semiconductor ban","travel ban","border","immigration",
    "asylum","deportation",
    # v5: generic policy verb — with a figure/actor it marks a policy series
    # ("Trump transgender sports ban?"), not a sports one.
    "ban","bans",
}

# POLITICS: with a NON-US ACTOR => direct (foreign politics). With US marker or no actor => background.
POLITICS = {
    "election","elections","president","presidential","prime minister","premier",
    "chancellor","parliament","referendum","vote","party leader","government","regime",
    "cabinet","minister","run for office","win the election","re-election","reelection",
    "nomination","primary","polls","ousted","overthrown",
}

# US markers — used by the US-border rule below.
US_MARKERS = {"united states","usa","america","american","u.s.","u.s.a.","white house"}

# BORDER_FAMILY: US marker + any of these => direct (rule 2 of old prompt: foreign policy,
# war, sanctions, defense, border). Alone => nothing.
BORDER_FAMILY = {
    "foreign policy","border","immigration","asylum","deportation","customs","tariff",
    "tariffs","trade war","trade deal","import ban","export ban","export controls",
    "chip ban","semiconductor ban","travel ban","visa","wall","defense budget",
    "military spending","arms sales","defense spending",
}

# CHANNEL: GPR-sensitive asset prices. Standalone => channel. Crypto deliberately NOT channel.
CHANNEL = {
    "crude oil","wti","brent","oil price","oil prices","oil futures","natural gas","lng",
    "gasoline","gas prices","fuel prices","heating oil","gold","silver","copper","platinum",
    "palladium","aluminum","nickel","uranium","wheat","corn","soybeans","fertilizer",
    "defense stocks","defense sector","aerospace","defense companies","lockheed",
    "raytheon","northrop","general dynamics","bae systems","freight rates","shipping rates",
    "container rates","baltic dry","baltice","shipping index","tanker rates","eur/usd",
    "usd/jpy","usd/cny","usd/rub","usd/uah","usd/ils","dollar index","dxy","ruble",
    "hryvnia","shekel","yuan","exchange rate","currency pair","safe haven","treasury yields",
    "10-year yield","30-year yield","gold price","oil price",
    # v3 — QA-verified: slash-free FX tickers, UST/treasury yield series that
    # evaded "10-year yield", gas price singular, and macro economics
    "usdjpy","usdcny","usdinr","usdbusd","usdeur","usdgbp","usdcad","usdmxn",
    "usdtry","usdzar","eurusd","gbpusd","ust 2 year","ust 5 year","ust 7 year",
    "ust 10 year","ust 20 year","ust 30 year","treasury 10y","treasury 30y",
    "treasury 2y","treasury 5y","treasury 7y","gas price","cpi","inflation",
    "inflation rate",
}

# MARKET_LEVEL_PATTERNS: strict underspecified-title templates only (see precedence note).
# The primary market-level mechanism is name-diversity (>=3 distinct normalized market
# names in the series), computed by classify.py; these templates cover the remaining
# generic-title cases.
MARKET_LEVEL_PATTERNS = (
    r"\bwhat will .+ (say|do|happen|occur|announce)\b",
    r"\bwhat (words|phrase|numbers|exact words|exact phrase|word|phrase)\b",
    r"\bwhich (of the following|of these|of the|among)\b",
    r"\b(one of|any of|none of) (the following|these)\b",
    r"\bwhat will happen\b",
    r"\bwhat (happens|comes) next\b",
    r"\bwhich (word|phrase|number|amount|date|time)\b",
)

# PATTERN_EXCLUDE: whole-title regexes (on normalized text) that are
# unambiguously non-geopolitical but not enumerable by term — v4:
# single-stock/ETF price-target bets ("Will PLTR hit (HIGH) $186 in August?").
# The `high` token pins it: "hit" alone would wrongly kill "Will Trump hit
# 40% approval?" (politics).
PATTERN_EXCLUDE = (
    r"\b[a-z]{1,6} hit high \d+",
)

# Precompiled matchers: terms are normalized exactly like the text (so
# `u.s.` becomes `u s`, `eur/usd` becomes `eur usd`) and matched as
# whole words (`\b…\b`) on the normalized text — the plan's stated rule,
# uniform for single- and multi-word terms.
def normalize(text: str) -> str:
    """NFKD → casefold → non-alphanumerics to single space → collapse/trim."""
    text = unicodedata.normalize("NFKD", text or "")
    folded = text.casefold()
    folded = re.sub(r"[^0-9a-z]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _compile_set(terms):
    return {t: re.compile(r"\b" + re.escape(normalize(t)) + r"\b") for t in terms}


_EXCLUDE_RX = _compile_set(EXCLUDE)
_ACTORS_RX = _compile_set(ACTORS)
_STRONG_RX = _compile_set(STRONG_ACTIONS)
_ACTIONS_RX = _compile_set(ACTIONS)
_POLITICS_RX = _compile_set(POLITICS)
_US_RX = _compile_set(US_MARKERS)
_BORDER_RX = _compile_set(BORDER_FAMILY)
_CHANNEL_RX = _compile_set(CHANNEL)
_PATTERN_RX = [re.compile(pat, re.IGNORECASE) for pat in MARKET_LEVEL_PATTERNS]
_PATTERN_EXCLUDE_RX = [re.compile(pat, re.IGNORECASE) for pat in PATTERN_EXCLUDE]


def _hits(text, matchers) -> list:
    """All terms from `matchers` present in `text` (whole-word, case-insensitive)."""
    found = []
    for term, rx in matchers.items():
        if rx.search(text):
            found.append(term)
    return found


def parse_tags(tags_json) -> list:
    """Tags arrive as a JSON string; tolerate anything the exchanges produce."""
    try:
        value = json.loads(tags_json or "[]")
    except (TypeError, ValueError):
        return []
    if isinstance(value, str):
        return [value]
    return [t for t in value if isinstance(t, str)] if isinstance(value, list) else []


def decide(title, category, tags_json, name_distinct=0):
    """Classify one series into (label, confidence, rationale).

    Strict rule order; the first firing rule wins:

    1. EXCLUDE hit in title/tags/category → background 0.85.
    2. CHANNEL hit → channel (title 0.95; tags/category-only 0.70).
    3. DIRECT, in order:
       - (ACTOR ∧ ACTIONS/STRONG_ACTIONS) both in title → 0.95.
       - STRONG_ACTIONS in title → 0.90.
       - STRONG_ACTIONS/`geopolitics`-family via tags/category only → 0.65;
         any other direct hit via tags/category only → 0.65.
         Tags/category-only direct hits are always below the default 0.7
         review threshold, so every one lands in the review queue. The
         title-driven rules above are unchanged.
       - (non-US ACTOR ∧ POLITICS) both in title → 0.90.
       - (US_MARKERS ∧ BORDER_FAMILY) both in title → 0.90.
    4. MARKET_LEVEL → 0.80 on name diversity (>=3 distinct market names),
       0.85 on a template pattern match.
    5. Default background → 0.55.

    `needs_review` is decided by the caller against a threshold, so this
    function stays a pure mapping from text to label.
    """
    norm_title = normalize(title or "")
    tags = parse_tags(tags_json)
    norm_tags = " ".join(normalize(t) for t in tags)
    norm_category = normalize(category or "")
    all_side = f"{norm_title} {norm_tags} {norm_category}"

    # 1. EXCLUDE — any hit anywhere, overrides every other rule.
    hits = _hits(all_side, _EXCLUDE_RX)
    if hits:
        side = "title" if _hits(norm_title, _EXCLUDE_RX) else "tags"
        return "background", 0.85, f"exclude: {', '.join(sorted(hits))} ({side})"
    for rx in _PATTERN_EXCLUDE_RX:
        if rx.search(norm_title):
            return "background", 0.85, f"exclude: {rx.pattern} (title)"

    # 2. CHANNEL.
    title_channel = _hits(norm_title, _CHANNEL_RX)
    if title_channel:
        return "channel", 0.95, f"channel: {', '.join(sorted(title_channel))}"
    side_channel = _hits(all_side, _CHANNEL_RX)
    if side_channel:
        return "channel", 0.70, f"channel: {', '.join(sorted(side_channel))} (tags)"

    # 3. DIRECT.
    # (actor ∧ action) both in title — action may be a STRONG_ACTIONS term too.
    title_actors = _hits(norm_title, _ACTORS_RX)
    title_actions = _hits(norm_title, _ACTIONS_RX)
    title_strong = _hits(norm_title, _STRONG_RX)
    if title_actors and (title_actions or title_strong):
        action = title_actions or title_strong
        return ("direct", 0.95,
                f"actor+action: {title_actors[0]}, {action[0]} (title)")

    # STRONG_ACTIONS in title.
    if title_strong:
        return "direct", 0.90, f"strong-action: {title_strong[0]} (title)"

    # Strong/geopolitics via tags or category only.
    side_strong = _hits(all_side, _STRONG_RX)
    if side_strong:
        return "direct", 0.65, f"strong-action: {side_strong[0]} (tags)"
    geo_tags = [t for t in tags if "geopolitic" in t.casefold()]
    if geo_tags:
        return "direct", 0.65, f"strong-action: {geo_tags[0]} (tags)"

    # Any other direct hit via tags/category only. Title-only ACTION hits are
    # NOT direct — ACTIONS require an ACTOR in the same text, so subtract the
    # terms already present in the title.
    side_text = f"{norm_tags} {norm_category}"
    side_hits = _hits(side_text, _ACTORS_RX) + _hits(side_text, _ACTIONS_RX)
    side_only = [t for t in side_hits
                 if t not in title_actors and t not in title_actions
                 and t not in title_strong]
    if side_only:
        return "direct", 0.65, f"actor+action: {', '.join(side_only)} (tags)"

    # (non-US ACTOR ∧ POLITICS) both in title → foreign politics.
    title_politics = _hits(norm_title, _POLITICS_RX)
    if title_actors and title_politics:
        return ("direct", 0.90,
                f"foreign-politics: {title_actors[0]}, {title_politics[0]} (title)")

    # (US_MARKERS ∧ BORDER_FAMILY) both in title → US-border.
    title_us = _hits(norm_title, _US_RX)
    title_border = _hits(norm_title, _BORDER_RX)
    if title_us and title_border:
        return ("direct", 0.90,
                f"us-border: {title_border[0]} (title)")

    # 4. MARKET_LEVEL.
    if name_distinct >= 3:
        return "market-level", 0.80, f"name-diversity: {name_distinct} distinct markets"
    for rx in _PATTERN_RX:
        if rx.search(norm_title):
            return "market-level", 0.85, f"template: {rx.pattern}"

    # 5. Default background.
    return "background", 0.55, "no evidence"
