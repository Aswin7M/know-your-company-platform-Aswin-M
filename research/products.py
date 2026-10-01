from __future__ import annotations

import re

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.product import Product, ProductList
from research.common import ModuleResult, ResearchContext, failed, finish, hit_ids


def _price_grounded(price: str, text_raw: str) -> bool:
    """A price is only kept if its figures literally appear in the cited evidence."""
    nums = [n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", price)]
    text = re.sub(r"(?<=\d),(?=\d)", "", text_raw)
    return bool(nums) and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text) for n in nums)


def extract_products(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("products services platform features solutions pricing integrations")
    if not hits:
        return ModuleResult([], ModuleStatus(status="partial", detail="No evidence retrieved for products."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="products", evidence=ctx.block(hits), schema_model=ProductList)
    if data is None:
        return failed("The model did not return valid JSON for products.")
    pool, items, dropped, seen = hit_ids(hits), [], 0, set()
    for raw in (data.get("products") or data.get("items") or [])[:10]:
        try:
            p = Product.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        sids = ctx.supporting_sources(p.product_name, p.source_ids, pool)
        if not sids or p.product_name.lower() in seen:
            dropped += 0 if p.product_name.lower() in seen else 1
            continue
        seen.add(p.product_name.lower())
        p.source_ids = sids
        text_norm, text_raw = ctx.joined_norm(sids), " ".join(ctx.text_raw(s) for s in sids)
        if p.description and ctx.overlap(p.description, text_norm) < 0.3:
            p.description = None
        if p.pricing and not _price_grounded(p.pricing, text_raw):
            p.pricing = None                      # pricing only if publicly verified
        items.append(p)
    return ModuleResult(items, finish("product(s)", items, dropped))
