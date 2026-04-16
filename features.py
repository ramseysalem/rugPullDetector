"""
features.py
-----------
Feature definitions and live extraction logic for rug pull detection.

All 18 features are computed from on-chain data fetched via TheGraph
(Uniswap V2 subgraph) and Ethplorer APIs.

Usage (live token prediction):
    from features import extract_features
    feat = extract_features(pair_id, token_id, token_index)
"""

import time
import requests
import json
from math import sqrt
from decimal import Decimal
from bs4 import BeautifulSoup
import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THEGRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
ETHPLORER_API_KEY = "EK-4L18F-Y2jC1b7-9qC3N"   # replace with your own key

# Addresses considered "burned" for token burn ratio calculation
BURN_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000001",
    "0x0000000000000000000000000000000000000002",
    "0x0000000000000000000000000000000000000003",
    "0x0000000000000000000000000000000000000004",
    "0x0000000000000000000000000000000000000005",
    "0x0000000000000000000000000000000000000006",
    "0x0000000000000000000000000000000000000007",
    "0x0000000000000000000000000000000000000008",
    "0x0000000000000000000000000000000000000009",
    "0x000000000000000000000000000000000000000a",
    "0x000000000000000000000000000000000000000b",
    "0x000000000000000000000000000000000000000c",
    "0x000000000000000000000000000000000000000d",
    "0x000000000000000000000000000000000000000e",
    "0x000000000000000000000000000000000000000f",
    "0x000000000000000000000000000000000000dead",
    "0x000000000000000000000000000000000000DEAD",
}

# Known LP locker contract addresses
LOCKER_ADDRESSES = {
    "0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214",
    "0xe2fe530c047f2d85298b07d9333c05737f1435fb",
    "0x000000000000000000000000000000000000dead",
}

# Proxy contracts — if creator matches these, fall back to first minter
PROXY_CONTRACTS = {
    "0x5e5a7b76462e4bdf83aa98795644281bdba80b88",
    "0x000000000092c287eb63e8c2c30b4a74787054f8",
    "0x0f4676178b5c53ae0a655f1b19a96387e4b8b5f2",
    "0xdf65f4e6f2e9436bc1de1e00661c7108290e8bd3",
    "0xdb73dde1867843fdca5244258f2fd4b6dc7b154e",
    "0xbdb1127bd15e76d7e4d3bc4f6c7801aa493e03f0",
    "0x8f84c1d37fa5e21c81a5bf4d3d5f2e718a2d8eb4",
    "0x908521c8e53e9bb3b8b9df51e2c6dd3079549382",
    "0x85aa7f78bdb2de8f3e0c0010d99ad5853ffcfc63",
    "0x909d05f384d0663ed4be59863815ab43b4f347ec",
    "0xb4a2810e9d0f1d4d2c0454789be80aaeb9188480",
    "0x96fc64f7fe4924546b9204fe22707e3df04be4c8",
    "0x226e390751a2e22449d611bac83bd267f2a2caff",
}


# ---------------------------------------------------------------------------
# Feature schema (for reference / documentation)
# ---------------------------------------------------------------------------

FEATURE_SCHEMA = {
    # --- LP / transaction-based features ---
    "mint_count_per_week": "Number of liquidity add (mint) events per week of active trading",
    "burn_count_per_week": "Number of liquidity remove (burn) events per week of active trading",
    "mint_ratio": "Mint events / total events (mint + swap + burn)",
    "swap_ratio": "Swap events / total events",
    "burn_ratio": "Burn events / total events",
    "mint_mean_period": "Average time of mint events relative to active period (normalised to [0,1])",
    "swap_mean_period": "Average time of swap events relative to active period (normalised to [0,1])",
    "burn_mean_period": "Average time of burn events relative to active period (normalised to [0,1])",
    "swap_in_per_week": "Buy-side swaps (ETH → token) per week",
    "swap_out_per_week": "Sell-side swaps (token → ETH) per week",
    "swap_rate": "swap_in / (swap_out + 1)  — buy pressure ratio",
    # --- LP holder features ---
    "lp_avg": "Average LP token share among top-100 holders (assuming equal split over significant holders)",
    "lp_std": "Standard deviation of LP token share",
    "lp_creator_holding_ratio": "Fraction of LP tokens held by the token creator",
    "lp_lock_ratio": "Fraction of LP tokens in a known locker contract",
    # --- Token holder features ---
    "token_burn_ratio": "Fraction of token supply sent to burn addresses",
    "token_creator_holding_ratio": "Fraction of token supply held by the creator",
    "number_of_token_creation_of_creator": "How many ERC-20 tokens this creator address has deployed (proxy for serial scammer)",
}


# ---------------------------------------------------------------------------
# TheGraph helpers
# ---------------------------------------------------------------------------

def _run_query(query: str) -> dict:
    resp = requests.post(THEGRAPH_URL, json={"query": query}, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"TheGraph query failed: {resp.status_code}")


def _get_mints(pair_id: str) -> list:
    template = '''
    { mints(first:1000, orderBy:timestamp, orderDirection:asc,
            where:{pair:"%s", timestamp_gt:%s}) {
        amount0 amount1 to sender timestamp } }
    '''
    results, ts = [], 0
    while True:
        data = _run_query(template % (pair_id, ts))["data"]["mints"]
        results.extend(data)
        if len(data) < 1000:
            break
        ts = data[-1]["timestamp"]
    return results


def _get_swaps(pair_id: str) -> list:
    template = '''
    { swaps(first:1000, orderBy:timestamp, orderDirection:asc,
            where:{pair:"%s", timestamp_gt:%s}) {
        amount0In amount0Out amount1In amount1Out to sender timestamp
        transaction { id } } }
    '''
    results, ts = [], 0
    while True:
        data = _run_query(template % (pair_id, ts))["data"]["swaps"]
        results.extend(data)
        if len(data) < 1000:
            break
        ts = data[-1]["timestamp"]
    return results


def _get_burns(pair_id: str) -> list:
    template = '''
    { burns(first:1000, orderBy:timestamp, orderDirection:asc,
            where:{pair:"%s", timestamp_gt:%s}) {
        amount0 amount1 to sender timestamp
        transaction { id } } }
    '''
    results, ts = [], 0
    while True:
        data = _run_query(template % (pair_id, ts))["data"]["burns"]
        results.extend(data)
        if len(data) < 1000:
            break
        ts = data[-1]["timestamp"]
    return results


# ---------------------------------------------------------------------------
# Ethplorer helpers
# ---------------------------------------------------------------------------

def _get_holders(address: str) -> list:
    url = (
        f"https://api.ethplorer.io/getTopTokenHolders/{address}"
        f"?apiKey={ETHPLORER_API_KEY}&limit=100"
    )
    resp = requests.get(url, timeout=20)
    if resp.status_code == 400:
        return []
    return resp.json().get("holders", [])


def _get_creator(pair_id: str, token_id: str) -> str:
    """Try Ethplorer → Etherscan scrape → first minter fallback."""
    url = f"https://api.ethplorer.io/getAddressInfo/{token_id}?apiKey={ETHPLORER_API_KEY}"
    try:
        data = requests.get(url, timeout=20).json()
        creator = data["contractInfo"]["creatorAddress"]
        if creator:
            if creator in PROXY_CONTRACTS:
                raise ValueError("proxy")
            return creator
    except Exception:
        pass

    # Etherscan scrape fallback
    try:
        resp = requests.get(
            f"https://etherscan.io/address/{token_id}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find("a", {"class": "hash-tag text-truncate"})
        creator = re.sub("<.+?>", "", str(tag), 0).strip()
        if creator and creator != "None":
            return creator
    except Exception:
        pass

    # First minter fallback
    query = '''{ mints(first:1, orderBy:timestamp, orderDirection:asc,
                       where:{pair:"%s"}) { to } }''' % pair_id
    return _run_query(query)["data"]["mints"][0]["to"]


# ---------------------------------------------------------------------------
# Individual feature computations
# ---------------------------------------------------------------------------

def _active_period(mints, swaps, burns) -> int:
    start = int(mints[0]["timestamp"])
    candidates = [mints[-1]["timestamp"]]
    if swaps:
        candidates.append(swaps[-1]["timestamp"])
    if burns:
        candidates.append(burns[-1]["timestamp"])
    return int(max(candidates)) - start


def _mean_period(txs, initial_ts: int) -> float:
    if not txs:
        return 0
    return sum(int(t["timestamp"]) - initial_ts for t in txs) / len(txs)


def _swap_io(swaps, eth_index: int) -> tuple:
    """Count buy-side vs sell-side swaps.  eth_index=0 means token0 is WETH."""
    swap_in = swap_out = 0
    if eth_index == 0:
        for s in swaps:
            if s["amount0In"] == "0":
                swap_out += 1
            else:
                swap_in += 1
    else:
        for s in swaps:
            if s["amount1In"] == "0":
                swap_out += 1
            else:
                swap_in += 1
    return swap_in, swap_out


def _lp_distribution(holders: list) -> tuple:
    """Return (avg_share, std_share) for holders with share >= 1%."""
    significant = [h for h in holders if h["share"] >= 1.0]
    n = len(significant) or 1
    avg = 100 / n
    std = sqrt(sum((h["share"] - avg) ** 2 for h in significant))
    return avg, std


def _lock_ratio(holders: list) -> float:
    for h in holders:
        if h["address"] in LOCKER_ADDRESSES:
            return h["share"]
    return 0.0


def _creator_share(holders: list, creator: str) -> float:
    for h in holders:
        if h["address"] == creator:
            return h["share"]
    return 0.0


def _burn_ratio(holders: list) -> float:
    for h in holders:
        if h["address"] in BURN_ADDRESSES:
            return h["share"]
    return 0.0


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_features(pair_id: str, token_id: str, eth_index: int) -> dict:
    """
    Fetch on-chain data and compute all 18 features for a single token pair.

    Parameters
    ----------
    pair_id   : Uniswap V2 pair address (hex)
    token_id  : ERC-20 token address (the non-WETH side)
    eth_index : 0 if token0 is WETH, 1 if token1 is WETH

    Returns
    -------
    dict with keys matching FEATURE_SCHEMA
    """
    mints = _get_mints(pair_id)
    swaps = _get_swaps(pair_id)
    burns = _get_burns(pair_id)

    initial_ts = int(mints[0]["timestamp"])
    period = _active_period(mints, swaps, burns) or 1

    weeks = (period / (60 * 60 * 24 * 7)) + 1

    mint_count = len(mints)
    swap_count = len(swaps)
    burn_count = len(burns)
    total_tx = mint_count + swap_count + burn_count or 1

    swap_in, swap_out = _swap_io(swaps, eth_index)

    creator = _get_creator(pair_id, token_id)

    lp_holders = _get_holders(pair_id)
    token_holders = _get_holders(token_id)

    lp_avg, lp_std = _lp_distribution(lp_holders)
    lp_lock = _lock_ratio(lp_holders)
    lp_creator = _creator_share(lp_holders, creator)

    # If lock expires within 3 days, treat as creator-held
    # (requires checking unlock date via Bitquery — omitted here for simplicity)

    token_burn = _burn_ratio(token_holders)
    token_creator = _creator_share(token_holders, creator)

    return {
        "mint_count_per_week": mint_count / weeks,
        "burn_count_per_week": burn_count / weeks,
        "mint_ratio": mint_count / total_tx,
        "swap_ratio": swap_count / total_tx,
        "burn_ratio": burn_count / total_tx,
        "mint_mean_period": _mean_period(mints, initial_ts) / period,
        "swap_mean_period": _mean_period(swaps, initial_ts) / period,
        "burn_mean_period": _mean_period(burns, initial_ts) / period,
        "swap_in_per_week": swap_in / weeks,
        "swap_out_per_week": swap_out / weeks,
        "swap_rate": swap_in / (swap_out + 1),
        "lp_avg": lp_avg,
        "lp_std": lp_std,
        "lp_creator_holding_ratio": lp_creator,
        "lp_lock_ratio": lp_lock,
        "token_burn_ratio": token_burn,
        "token_creator_holding_ratio": token_creator,
        "number_of_token_creation_of_creator": 1,  # requires separate lookup
    }


if __name__ == "__main__":
    print("Feature schema:")
    for name, desc in FEATURE_SCHEMA.items():
        print(f"  {name:<45} {desc}")
