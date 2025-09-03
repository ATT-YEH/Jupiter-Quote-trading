#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pool→Pool Arbitrage (paper)
- 僅在「指定池 A」做 USDC→TOKEN，再在「指定池 B」做 TOKEN→USDC
- 兩腿都用 Jupiter v6 /quote，強制 onlyDirectRoutes，且 route 的 ammKey 必須等於你指定的池地址
- 達到門檻(PERSIST_MS)、通過冷卻(COOLDOWN_S)才觸發紙上成交
- 抗 429、支援 tokens.jup.ag 跳轉、CSV 紀錄

pip install httpx python-dotenv
"""

import os, sys, time, csv, math, random, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

# ---------- env ----------
ENV = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV)

def _sz(s: str | None) -> str:
    return (s or "").replace("<","").replace(">","").strip()

# 基準/標的
MINT_IN     = _sz(os.getenv("MINT_IN"))       # 建議用原生 USDC: EPjF...
MINT_TOKEN  = _sz(os.getenv("MINT_TOKEN"))    # 你的 meme 幣 mint
DEC_IN      = os.getenv("DEC_IN", "6")        # 寫死 6 最保險（USDC）

# 指定兩個池（Dexscreener 的 Pair address）
POOL_A_AMM  = _sz(os.getenv("POOL_A_AMM"))    # 便宜池（買入）
POOL_B_AMM  = _sz(os.getenv("POOL_B_AMM"))    # 貴池（賣出）
# 可選：限制 DEX（幫助 Jupiter 挑正確池；常見：Meteora、RaydiumClmm、OrcaWhirlpool）
POOL_A_DEX  = _sz(os.getenv("POOL_A_DEX", ""))
POOL_B_DEX  = _sz(os.getenv("POOL_B_DEX", ""))

# 交易參數
CANDIDATE_USD_SIZES = [float(x) for x in os.getenv("CANDIDATE_USD_SIZES","0.5,1,2,5,10").split(",")]
SLIPPAGE_BPS  = int(os.getenv("SLIPPAGE_BPS","80"))
MIN_EDGE_BPS  = float(os.getenv("MIN_EDGE_BPS","60"))
FLAT_COST_USD = float(os.getenv("FLAT_COST_USD","0.01"))
PERSIST_MS    = int(os.getenv("PERSIST_MS","1000"))
COOLDOWN_S    = float(os.getenv("COOLDOWN_S","3"))
POLL_MS       = int(os.getenv("POLL_MS","1500"))
SYMBOL        = os.getenv("SYMBOL","TOKEN")

# 額外費（一般設 0；Jupiter 已含池費）
FEE_BUY_BPS   = float(os.getenv("FEE_BUY_BPS","0"))
FEE_SELL_BPS  = float(os.getenv("FEE_SELL_BPS","0"))

# HTTP 限流
RETRY_429       = int(os.getenv("RETRY_429","4"))
BACKOFF_BASE_MS = int(os.getenv("BACKOFF_BASE_MS","600"))
JITTER_MS       = int(os.getenv("JITTER_MS","300"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_S","6.0"))
USER_AGENT      = os.getenv("USER_AGENT","pool2pool-jup/1.0")
DEBUG           = os.getenv("DEBUG_PRINT_IDS","true").lower()=="true"

# 基本檢查
def _looks_like_mint(s:str)->bool:
    return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", s or ""))

for name,val in [("MINT_IN",MINT_IN),("MINT_TOKEN",MINT_TOKEN),("POOL_A_AMM",POOL_A_AMM),("POOL_B_AMM",POOL_B_AMM)]:
    if not _looks_like_mint(val):
        print(f"[CONFIG] {name} 看起來不是有效的 Solana 位址：{val}")
        sys.exit(1)

if DEBUG:
    print("MINT_IN   =", MINT_IN)
    print("MINT_TOKEN=", MINT_TOKEN)
    print("POOL_A_AMM=", POOL_A_AMM, "| DEX:", POOL_A_DEX or "(any)")
    print("POOL_B_AMM=", POOL_B_AMM, "| DEX:", POOL_B_DEX or "(any)")

# ---------- HTTP client ----------
client = httpx.Client(http2=True, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})

JUP = "https://quote-api.jup.ag"
QUOTE = f"{JUP}/v6/quote"

# tokens list（只用來保險抓小數；USDC 直接 DEC_IN=6 就行）
TOKENS_URLS = ["https://tokens.jup.ag/tokens","https://token.jup.ag/all"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def backoff(attempt:int, retry_after:float|None=None):
    if retry_after and retry_after>0:
        time.sleep(retry_after); return
    ms = BACKOFF_BASE_MS*(2**attempt) + random.uniform(0,JITTER_MS)
    time.sleep(ms/1000.0)

def fetch_json(url:str)->dict|list:
    for k in range(RETRY_429+1):
        try:
            r = client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code==429:
                ra=e.response.headers.get("Retry-After")
                s=float(ra) if (ra and ra.replace(".","",1).isdigit()) else None
                print(f"[WARN] 429 {url} attempt {k+1}"); backoff(k,s); continue
            print("[ERROR] HTTP:", e.response.status_code, "body:", e.response.text[:200])
            raise
        except httpx.RequestError as e:
            print("[WARN] net err:", e); backoff(k); continue
    raise RuntimeError(f"Exceeded retries: {url}")

_token_list: List[dict] | None = None
def get_decimals(mint:str, default:int=6)->int:
    # 先用 env 覆寫
    if mint==MINT_IN and DEC_IN.isdigit():
        return int(DEC_IN)
    global _token_list
    if _token_list is None:
        for u in TOKENS_URLS:
            try:
                data = fetch_json(u)
                _token_list = data["tokens"] if isinstance(data,dict) and "tokens" in data else data
                break
            except Exception as e:
                print(f"[WARN] token list load fail from {u}: {e}")
                _token_list=[]
    for t in _token_list:
        addr = t.get("address") or t.get("mint")
        if addr==mint:
            try: return int(t.get("decimals", default))
            except: return default
    print(f"[WARN] decimals not found for {mint}, use {default}")
    return default

def to_atomic(amount:float, decimals:int)->int:
    return int(round(amount * (10**decimals)))

def quote_on_exact_amm(input_mint:str, output_mint:str, in_amount:int, slippage_bps:int, required_amm:str, dex_filter:str="")->dict|None:
    """
    向 Jupiter 詢價；僅接受:
      - onlyDirectRoutes=true（單跳）
      - routePlan[0].swapInfo.ammKey == required_amm
      - 若 dex_filter 非空，附加 &dexes=dex_filter 幫助命中
    命中則回傳整個 JSON（含 outAmount/priceImpactPct/routePlan），否則回傳 None
    """
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(in_amount),
        "slippageBps": str(slippage_bps),
        "swapMode": "ExactIn",
        "onlyDirectRoutes": "true",
    }
    if dex_filter:
        params["dexes"] = dex_filter

    for k in range(RETRY_429+1):
        try:
            r = client.get(QUOTE, params=params)
            r.raise_for_status()
            data = r.json()
            plan = data.get("routePlan") or []
            if not plan:
                return None
            hop = plan[0].get("swapInfo", {})
            amm_key = hop.get("ammKey") or ""
            if amm_key == required_amm:
                return data
            # 沒命中指定池
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code==429:
                ra=e.response.headers.get("Retry-After")
                s=float(ra) if (ra and ra.replace(".","",1).isdigit()) else None
                print(f"[WARN] 429 quote attempt {k+1}"); backoff(k,s); continue
            print("[ERROR] quote", e.response.status_code, e.response.text[:200])
            return None
        except httpx.RequestError as e:
            print("[WARN] net err on quote:", e); backoff(k); continue
    return None

# ---------- CSV ----------
LOG_DIR = os.path.abspath("./logs"); os.makedirs(LOG_DIR, exist_ok=True)
TRADES_CSV = os.path.join(LOG_DIR, "paper_trades.csv")
QUOTES_CSV = os.path.join(LOG_DIR, "quotes.csv")

if not os.path.exists(TRADES_CSV):
    with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts","symbol","size_usd",
            "leg1_pool","leg2_pool",
            "leg1_out_token_atomic","leg2_out_usdc",
            "raw_bps","fees_bps","net_bps",
            "edge_per_usd","min_break_even_usd","decided_usd",
            "slippage_bps","flat_cost_usd","pnl_usd","notes"
        ])
if not os.path.exists(QUOTES_CSV):
    with open(QUOTES_CSV,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts","symbol","size_usd",
            "q1_out_token_atomic","q1_priceImpactPct",
            "q2_out_usdc_atomic","q2_priceImpactPct"
        ])

# ---------- state ----------
_last_trade_ts = 0.0
_edge_hold_ms = 0

def write_trade(**kw):
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            kw.get("ts"), kw.get("symbol"), f"{kw.get('size_usd',0):.4f}",
            kw.get("leg1_pool",""), kw.get("leg2_pool",""),
            kw.get("leg1_out_token_atomic",0), f"{kw.get('leg2_out_usdc',0.0):.6f}",
            f"{kw.get('raw_bps',0.0):.2f}", f"{kw.get('fees_bps',0.0):.2f}", f"{kw.get('net_bps',0.0):.2f}",
            f"{kw.get('edge_per_usd',0.0):.6f}",
            f"{kw.get('min_break_even_usd',float('inf')) if math.isfinite(kw.get('min_break_even_usd',float('inf'))) else float('inf'):.6f}",
            f"{kw.get('decided_usd',0.0):.2f}",
            kw.get("slippage_bps",0), f"{kw.get('flat_cost_usd',0.0):.4f}",
            f"{kw.get('pnl_usd',0.0):.6f}", ";".join(kw.get("notes",[]))
        ])

def write_quote(**kw):
    with open(QUOTES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            kw.get("ts"), kw.get("symbol"), f"{kw.get('size_usd',0):.4f}",
            kw.get("q1_out_token_atomic",0), f"{kw.get('pi1',0.0):.6f}",
            kw.get("q2_out_usdc_atomic",0), f"{kw.get('pi2',0.0):.6f}"
        ])

def loop_once():
    global _last_trade_ts, _edge_hold_ms

    in_dec = int(DEC_IN) if DEC_IN.isdigit() else get_decimals(MINT_IN, 6)

    for size_usd in CANDIDATE_USD_SIZES:
        in_atomic = int(round(size_usd * (10**in_dec)))

        # leg1: 必須走 POOL_A_AMM
        q1 = quote_on_exact_amm(MINT_IN, MINT_TOKEN, in_atomic, SLIPPAGE_BPS, POOL_A_AMM, POOL_A_DEX)
        if not q1:
            print(f"[{now_iso()}] {SYMBOL} ${size_usd:.2f} | no_route_on_pool_A")
            write_quote(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                        q1_out_token_atomic=0, q2_out_usdc_atomic=0, pi1=0.0, pi2=0.0)
            write_trade(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                        leg1_pool=POOL_A_AMM, leg2_pool=POOL_B_AMM,
                        leg1_out_token_atomic=0, leg2_out_usdc=0.0,
                        raw_bps=0.0, fees_bps=0.0, net_bps=0.0,
                        edge_per_usd=0.0, min_break_even_usd=float("inf"),
                        decided_usd=0.0, slippage_bps=SLIPPAGE_BPS,
                        flat_cost_usd=FLAT_COST_USD, pnl_usd=0.0,
                        notes=["no_route_pool_A"])
            continue

        out_token_atomic = int(q1.get("outAmount", 0))
        pi1 = float(q1.get("priceImpactPct", 0.0) or 0.0)
        if out_token_atomic <= 0:
            print(f"[{now_iso()}] {SYMBOL} ${size_usd:.2f} | zero_out_on_A")
            continue

        # leg2: 必須走 POOL_B_AMM
        q2 = quote_on_exact_amm(MINT_TOKEN, MINT_IN, out_token_atomic, SLIPPAGE_BPS, POOL_B_AMM, POOL_B_DEX)
        if not q2:
            print(f"[{now_iso()}] {SYMBOL} ${size_usd:.2f} | no_route_on_pool_B")
            write_quote(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                        q1_out_token_atomic=out_token_atomic, q2_out_usdc_atomic=0, pi1=pi1, pi2=0.0)
            write_trade(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                        leg1_pool=POOL_A_AMM, leg2_pool=POOL_B_AMM,
                        leg1_out_token_atomic=out_token_atomic, leg2_out_usdc=0.0,
                        raw_bps=0.0, fees_bps=0.0, net_bps=0.0,
                        edge_per_usd=0.0, min_break_even_usd=float("inf"),
                        decided_usd=0.0, slippage_bps=SLIPPAGE_BPS,
                        flat_cost_usd=FLAT_COST_USD, pnl_usd=0.0,
                        notes=["no_route_pool_B"])
            continue

        out_usdc_atomic = int(q2.get("outAmount", 0))
        pi2 = float(q2.get("priceImpactPct", 0.0) or 0.0)

        out_usd = out_usdc_atomic / (10**in_dec)
        in_usd  = size_usd

        fees_bps = FEE_BUY_BPS + FEE_SELL_BPS
        raw_bps = 10000.0 * ((out_usd - in_usd) / ((out_usd + in_usd)/2.0)) if (out_usd+in_usd)>0 else -1e9
        net_bps = raw_bps - fees_bps

        edge_per_usd = (out_usd - in_usd) / max(in_usd,1e-9)
        min_break_even_usd = FLAT_COST_USD / max(edge_per_usd,1e-12) if edge_per_usd>0 else float("inf")

        notes = []
        nowt = time.time()
        if net_bps >= MIN_EDGE_BPS: _edge_hold_ms += POLL_MS
        else: _edge_hold_ms = 0
        persist_ok  = (PERSIST_MS<=0) or (_edge_hold_ms>=PERSIST_MS)
        cooldown_ok = (COOLDOWN_S<=0) or ((nowt - _last_trade_ts)>=COOLDOWN_S)

        if not cooldown_ok: notes.append("cooldown")
        if not persist_ok:  notes.append("waiting_persist")
        if not math.isfinite(min_break_even_usd): notes.append("edge<=0")

        should = (net_bps>=MIN_EDGE_BPS) and persist_ok and cooldown_ok and math.isfinite(min_break_even_usd)

        if should and (size_usd >= min_break_even_usd):
            decided_usd = size_usd
            pnl_usd = (out_usd - in_usd) - FLAT_COST_USD
            _last_trade_ts = nowt
            _edge_hold_ms = 0
        else:
            decided_usd = 0.0
            pnl_usd = 0.0
            if should and size_usd < min_break_even_usd: notes.append("size<minN")

        write_quote(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                    q1_out_token_atomic=out_token_atomic, q2_out_usdc_atomic=out_usdc_atomic,
                    pi1=pi1, pi2=pi2)

        write_trade(ts=now_iso(), symbol=SYMBOL, size_usd=size_usd,
                    leg1_pool=POOL_A_AMM, leg2_pool=POOL_B_AMM,
                    leg1_out_token_atomic=out_token_atomic, leg2_out_usdc=out_usd,
                    raw_bps=raw_bps, fees_bps=fees_bps, net_bps=net_bps,
                    edge_per_usd=edge_per_usd, min_break_even_usd=min_break_even_usd,
                    decided_usd=decided_usd, slippage_bps=SLIPPAGE_BPS,
                    flat_cost_usd=FLAT_COST_USD, pnl_usd=pnl_usd, notes=notes)

        print(f"[{now_iso()}] {SYMBOL} ${size_usd:.2f} | net {net_bps:.2f} bps | "
              f"minN ${min_break_even_usd if math.isfinite(min_break_even_usd) else float('inf'):.2f} | "
              f"decide ${decided_usd:.2f} pnl ${pnl_usd:.4f} "
              f"{'[' + ','.join(notes) + ']' if notes else ''}")

def main():
    print("\n== Pool→Pool Jupiter Paper Trader ==")
    print(f"A: {POOL_A_AMM} ({POOL_A_DEX or 'any'})  ->  B: {POOL_B_AMM} ({POOL_B_DEX or 'any'})")
    print(f"Pair: {MINT_IN} -> {MINT_TOKEN} -> {MINT_IN}")
    print(f"Sizes: {CANDIDATE_USD_SIZES} | slippage={SLIPPAGE_BPS} bps | min_edge={MIN_EDGE_BPS} bps")
    print(f"Persist: {PERSIST_MS} ms | Cooldown: {COOLDOWN_S} s | Poll: {POLL_MS} ms")
    print(f"Logs -> {os.path.abspath('./logs')}\n")
    try:
        while True:
            start = time.time()
            try:
                loop_once()
            except Exception as e:
                print(f"[WARN] loop error: {e}")
            time.sleep(max(0.0, POLL_MS/1000.0 - (time.time()-start)))
    except KeyboardInterrupt:
        print("\nBye.")

if __name__ == "__main__":
    main()
