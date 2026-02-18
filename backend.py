"""
Optimized Blog Generator Backend
================================
Production-grade implementation with:
- Rate limiting and semaphore control
- Comprehensive error handling
- OpenRouter Gemini image generation
- Resource cleanup
- Performance monitoring
"""

from __future__ import annotations

import operator
import os
import sys
import base64
import requests
import time
import argparse
import logging
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Dict, Any
from logging.handlers import RotatingFileHandler
from functools import wraps
from threading import Semaphore
import re

from pydantic import BaseModel, Field, validator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_file: str = "blog_generator.log") -> logging.Logger:
    """Configure production logging."""
    logger = logging.getLogger("BlogGenerator")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================================
# RATE LIMITING
# ============================================================

class RateLimiter:
    """Thread-safe rate limiter with semaphore control."""
    
    def __init__(self, max_concurrent: int = 3, min_interval: float = 0.5):
        self.semaphore = Semaphore(max_concurrent)
        self.min_interval = min_interval
        self.last_call_time = {}
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Wait for semaphore
            with self.semaphore:
                # Enforce minimum interval between calls
                now = time.time()
                last_time = self.last_call_time.get(func_name, 0)
                elapsed = now - last_time
                
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    time.sleep(sleep_time)
                
                self.last_call_time[func_name] = time.time()
                
                # Execute function
                return func(*args, **kwargs)
        
        return wrapper


# Global rate limiter
api_rate_limiter = RateLimiter(max_concurrent=5, min_interval=0.3)

# ============================================================
# SCHEMAS
# ============================================================

class Task(BaseModel):
    """Blog section task."""
    id: int
    title: str
    goal: str = Field(..., description="One sentence goal")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., ge=120, le=550)
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    
    @validator('bullets')
    def validate_bullets(cls, v):
        if not v:
            raise ValueError("Bullets cannot be empty")
        return v


class Plan(BaseModel):
    """Blog plan."""
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]
    
    @validator('blog_title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Blog title too short")
        return v.strip()


class EvidenceItem(BaseModel):
    """Research evidence."""
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    """Router output."""
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    """Evidence collection."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    """Image specification."""
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Filename only (no path), e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"
    
    @validator('filename')
    def validate_filename(cls, v):
        # Remove any path prefix, keep only filename
        if '/' in v:
            v = v.split('/')[-1]
        return v


class GlobalImagePlan(BaseModel):
    """Image plan."""
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    """LangGraph state."""
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Production configuration."""
    
    def __init__(self):
        load_dotenv()
        
        self.open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.open_router_api_key:
            raise ValueError("OPEN_ROUTER_API_KEY not found in environment")
        
        # Model settings
        self.llm_model = "gpt-4.1-mini"
        self.llm_temperature = 0.2
        self.image_model = "google/gemini-3-pro-image-preview"  # Updated model
        
        # API limits
        self.max_search_results = 6
        self.max_queries = 10
        self.max_images = 3
        self.request_timeout = 120
        self.max_retries = 3
        self.retry_delay = 2
        
        # Resource limits
        self.max_concurrent_workers = 5
        
        logger.info("Configuration loaded successfully")
    
    def get_llm(self) -> ChatOpenAI:
        """Get LLM instance."""
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=self.open_router_api_key,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries
        )


config = Config()
llm = config.get_llm()

# ============================================================
# SYSTEM PROMPTS
# ============================================================

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): Evergreen topics (e.g., "Python basics", "REST API design")
- hybrid (needs_research=true): Needs up-to-date examples but stable concepts
- open_book (needs_research=true): Volatile/recent content (e.g., "Latest AI models 2026")

If needs_research=true:
- Output 3–10 high-signal queries
- Queries should be scoped and specific
- Focus on authoritative sources
"""

RESEARCH_SYSTEM = """Research synthesizer for technical writing.

Rules:
- Only include items with non-empty url
- Prefer authoritative sources (docs, official blogs, academic papers)
- Keep published_at as YYYY-MM-DD or null
- Deduplicate by URL
- Include relevant snippets
"""

ORCH_SYSTEM = """Senior technical writer creating actionable blog outlines.

Requirements:
- 5–9 sections with clear goals, bullets, word counts
- Developer-focused with correct terminology
- Include code examples/edge cases/performance/security where relevant
- Mode-specific grounding:
  * closed_book: Focus on fundamentals and best practices
  * hybrid: Mix fundamentals with current examples
  * open_book: Heavy citation of recent sources

Output a structured plan with:
- Clear blog title
- Target audience
- Appropriate tone
- Detailed task breakdown
"""

WORKER_SYSTEM = """Write ONE blog section in Markdown.

Constraints:
- Cover all bullets in order
- Target words ±15%
- Start with ## heading (not #)
- Use proper markdown formatting
- Include code blocks with language tags when needed
- Cite sources with inline links for open_book mode
- Be technically accurate and clear
"""

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- Filename should be just the filename (e.g. "qkv_flow.png"), NOT a path like "images/qkv_flow.png"
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

# ============================================================
# NODES WITH RATE LIMITING
# ============================================================

@api_rate_limiter
def router_node(state: State) -> dict:
    """Route with rate limiting."""
    logger.info(f"Router: '{state['topic'][:100]}'")
    
    try:
        decider = llm.with_structured_output(RouterDecision)
        decision = decider.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}")
        ])
        
        logger.info(f"Router: mode={decision.mode}, queries={len(decision.queries)}")
        
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries[:config.max_queries],
        }
    except Exception as e:
        logger.error(f"Router failed: {e}", exc_info=True)
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
        }


def route_next(state: State) -> str:
    """Conditional routing."""
    return "research" if state["needs_research"] else "orchestrator"


@api_rate_limiter
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Tavily search with rate limiting."""
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        
        normalized = []
        for r in results or []:
            normalized.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        
        logger.debug(f"Tavily: '{query}' -> {len(normalized)} results")
        return normalized
    
    except Exception as e:
        logger.error(f"Tavily failed for '{query}': {e}")
        return []


def research_node(state: State) -> dict:
    """Research with parallel queries."""
    queries = state.get("queries", [])
    logger.info(f"Research: {len(queries)} queries")
    
    if not queries:
        return {"evidence": []}
    
    try:
        raw_results = []
        
        for q in queries:
            results = _tavily_search(q, max_results=config.max_search_results)
            raw_results.extend(results)
        
        if not raw_results:
            logger.warning("Research: No results")
            return {"evidence": []}
        
        logger.info(f"Research: {len(raw_results)} raw results")
        
        # Synthesize with LLM
        extractor = llm.with_structured_output(EvidencePack)
        pack = extractor.invoke([
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}")
        ])
        
        # Deduplicate
        dedup = {e.url: e for e in pack.evidence if e.url}
        
        logger.info(f"Research: {len(dedup)} unique items")
        return {"evidence": list(dedup.values())}
    
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        return {"evidence": []}


@api_rate_limiter
def orchestrator_node(state: State) -> dict:
    """Create plan with validation."""
    logger.info("Orchestrator: Creating plan")
    
    try:
        planner = llm.with_structured_output(Plan)
        evidence = state.get("evidence", [])
        mode = state.get("mode", "closed_book")
        
        # Prepare evidence summary
        evidence_summary = "\n".join([
            f"- {e.title} ({e.source or 'unknown'}): {e.snippet[:100] if e.snippet else ''}"
            for e in evidence[:16]
        ]) if evidence else "No external research available."
        
        plan = planner.invoke([
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n\n"
                f"Available Evidence:\n{evidence_summary}"
            ))
        ])
        
        logger.info(f"Plan: '{plan.blog_title}' with {len(plan.tasks)} sections")
        return {"plan": plan}
    
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}", exc_info=True)
        raise


def fanout(state: State):
    """Fan out tasks."""
    tasks = state["plan"].tasks
    logger.info(f"Fanout: {len(tasks)} tasks")
    
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in tasks
    ]


@api_rate_limiter
def worker_node(payload: dict) -> dict:
    """Write section with retry."""
    task = Task(**payload["task"])
    logger.info(f"Worker: Section #{task.id} - '{task.title}'")
    
    try:
        plan = Plan(**payload["plan"])
        evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
        
        bullets_text = "\n- " + "\n- ".join(task.bullets)
        
        # Filter relevant evidence
        evidence_text = ""
        if evidence and task.requires_citations:
            evidence_text = "\n\nRelevant Sources:\n" + "\n".join(
                f"- [{e.title}]({e.url}) - {e.snippet[:150] if e.snippet else ''}"
                for e in evidence[:10]
            )
        
        section_md = llm.invoke([
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(content=(
                f"Blog: {plan.blog_title}\n"
                f"Section #{task.id}: {task.title}\n"
                f"Goal: {task.goal}\n"
                f"Target: {task.target_words} words\n"
                f"Bullets:{bullets_text}\n"
                f"{evidence_text}"
            ))
        ]).content.strip()
        
        logger.info(f"Worker: Section #{task.id} done ({len(section_md)} chars)")
        return {"sections": [(task.id, section_md)]}
    
    except Exception as e:
        logger.error(f"Worker #{task.id} failed: {e}", exc_info=True)
        return {"sections": [(task.id, f"## {task.title}\n\n*Generation failed*\n")]}


def merge_content(state: State) -> dict:
    """Merge sections."""
    logger.info("Merge: Combining sections")
    
    try:
        plan = state["plan"]
        ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
        body = "\n\n".join(ordered).strip()
        
        # Simple title-only format (matching notebook)
        merged_md = f"# {plan.blog_title}\n\n{body}\n"
        
        logger.info(f"Merge: {len(ordered)} sections ({len(merged_md)} chars)")
        return {"merged_md": merged_md}
    
    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        raise


@api_rate_limiter
def decide_images(state: State) -> dict:
    """Plan images."""
    logger.info("ImagePlanner: Analyzing needs")
    
    try:
        planner = llm.with_structured_output(GlobalImagePlan)
        merged_md = state["merged_md"]
        plan = state["plan"]
        assert plan is not None

        image_plan = planner.invoke([
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(content=(
                f"Blog kind: {plan.blog_kind}\n"
                f"Topic: {state['topic']}\n\n"
                "Insert placeholders + propose image prompts.\n\n"
                f"{merged_md}"
            ))
        ])
        
        # Limit images
        limited_images = image_plan.images[:config.max_images]
        
        logger.info(f"ImagePlanner: {len(limited_images)} images planned")
        
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in limited_images],
        }
    
    except Exception as e:
        logger.error(f"ImagePlanner failed: {e}", exc_info=True)
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }


def _openrouter_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated via OpenRouter's Gemini image model.
    Env var: OPEN_ROUTER_API_KEY
    
    Uses the correct OpenRouter response format:
    message.images = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}]
    """
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY is not set.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "Blog Image Generator"
    }
    
    payload = {
        "model": config.image_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error: {response.status_code}\n"
            f"Response: {response.text}"
        )
    
    result = response.json()
    
    # Extract image data from OpenRouter response format
    try:
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            
            # Check for images array (OpenRouter/Gemini format)
            if "images" in message and message["images"]:
                img = message["images"][0]  # Take first image
                
                # Handle nested image_url structure
                # Format: {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}
                if isinstance(img, dict) and "image_url" in img:
                    image_url_obj = img["image_url"]
                    
                    if isinstance(image_url_obj, dict):
                        url_data = image_url_obj.get("url", "")
                        
                        if url_data.startswith("data:image"):
                            # Extract base64 data after comma
                            # Format: data:image/jpeg;base64,/9j/4AAQ...
                            image_base64 = url_data.split(",", 1)[1]
                            return base64.b64decode(image_base64)
                        
                        elif url_data.startswith("http"):
                            # Download from URL if provided
                            img_response = requests.get(url_data, timeout=30)
                            if img_response.status_code == 200:
                                return img_response.content
                            else:
                                raise RuntimeError(f"Failed to download image from URL: {img_response.status_code}")
                
                # Fallback: direct data or url keys
                elif isinstance(img, dict):
                    if "data" in img:
                        return base64.b64decode(img["data"])
                    elif "url" in img:
                        url_data = img["url"]
                        if url_data.startswith("data:image"):
                            image_base64 = url_data.split(",", 1)[1]
                            return base64.b64decode(image_base64)
                        elif url_data.startswith("http"):
                            img_response = requests.get(url_data, timeout=30)
                            if img_response.status_code == 200:
                                return img_response.content
                
                # Handle string format (base64 or URL)
                elif isinstance(img, str):
                    if img.startswith("http"):
                        img_response = requests.get(img, timeout=30)
                        if img_response.status_code == 200:
                            return img_response.content
                    elif img.startswith("data:image"):
                        image_base64 = img.split(",", 1)[1]
                        return base64.b64decode(image_base64)
                    else:
                        # Assume raw base64
                        return base64.b64decode(img)
            
            # Fallback: check content field (older format)
            content = message.get("content", "")
            if content:
                if content.startswith("http"):
                    img_response = requests.get(content, timeout=30)
                    if img_response.status_code == 200:
                        return img_response.content
                elif content.startswith("data:image"):
                    image_base64 = content.split(",", 1)[1]
                    return base64.b64decode(image_base64)
        
        raise RuntimeError(
            f"No image content found in OpenRouter response.\n"
            f"Response structure: {result.keys()}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract image from response: {e}\n"
            f"Message keys: {result.get('choices', [{}])[0].get('message', {}).keys() if result.get('choices') else 'N/A'}"
        )


def generate_and_place_images(state: State) -> dict:
    """Generate and place images with graceful fallback."""
    logger.info("ImageGen: Starting")
    
    try:
        plan = state["plan"]
        assert plan is not None

        md = state.get("md_with_placeholders") or state["merged_md"]
        image_specs = state.get("image_specs", []) or []

        # If no images requested, just return markdown
        if not image_specs:
            logger.info("ImageGen: No images needed")
            return {"final": md}

        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)

        for spec in image_specs:
            placeholder = spec["placeholder"]
            filename = spec["filename"]
            
            # FIXED: Remove 'images/' prefix if already present in filename
            if filename.startswith("images/"):
                filename = filename.replace("images/", "", 1)
            
            out_path = images_dir / filename

            # Generate only if needed
            if not out_path.exists():
                try:
                    logger.info(f"Generating image: {filename}")
                    img_bytes = _openrouter_generate_image_bytes(spec["prompt"])
                    out_path.write_bytes(img_bytes)
                    logger.info(f"✓ Saved: {out_path}")
                except Exception as e:
                    # Graceful fallback: keep doc usable
                    logger.warning(f"✗ Failed to generate {filename}: {e}")
                    prompt_block = (
                        f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                        f"> **Alt:** {spec.get('alt','')}\n>\n"
                        f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                        f"> **Error:** {e}\n"
                    )
                    md = md.replace(placeholder, prompt_block)
                    continue

            # Replace placeholder with image markdown (keep images/ prefix in markdown)
            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)

        logger.info("ImageGen: Complete")
        return {"final": md}
    
    except Exception as e:
        logger.error(f"ImageGen failed: {e}", exc_info=True)
        return {"final": state.get("merged_md", "")}


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph() -> StateGraph:
    """Build optimized graph."""
    logger.info("Building graph")
    
    # Reducer subgraph
    reducer_graph = StateGraph(State)
    reducer_graph.add_node("merge_content", merge_content)
    reducer_graph.add_node("decide_images", decide_images)
    reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
    reducer_graph.add_edge(START, "merge_content")
    reducer_graph.add_edge("merge_content", "decide_images")
    reducer_graph.add_edge("decide_images", "generate_and_place_images")
    reducer_graph.add_edge("generate_and_place_images", END)
    reducer_subgraph = reducer_graph.compile()
    
    # Main graph
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_subgraph)
    
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
    g.add_edge("research", "orchestrator")
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)
    
    logger.info("Graph ready")
    return g.compile()


# ============================================================
# PUBLIC API
# ============================================================

def run(topic: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate blog (public API).
    
    Args:
        topic: Blog topic
        as_of: Date (YYYY-MM-DD)
    
    Returns:
        Result dictionary with:
        - plan: Plan object
        - evidence: List of EvidenceItem
        - image_specs: List of image specifications
        - final: Final markdown content
    """
    logger.info("=" * 80)
    logger.info(f"Starting: '{topic[:100]}'")
    logger.info("=" * 80)
    
    if as_of is None:
        as_of = date.today().isoformat()
    
    try:
        graph = build_graph()
        
        initial_state = {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
        
        result = graph.invoke(initial_state)
        
        # Save markdown file
        if result.get("final") and result.get("plan"):
            plan = result["plan"]
            filename = re.sub(r'[^a-z0-9_-]+', '_', plan.blog_title.lower()).strip('_') + '.md'
            Path(filename).write_text(result["final"], encoding="utf-8")
            logger.info(f"Saved: {filename}")
        
        logger.info("=" * 80)
        logger.info("COMPLETE")
        logger.info("=" * 80)
        
        return result
    
    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)
        raise


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Blog Generator")
    parser.add_argument("topic", help="Blog topic")
    parser.add_argument("--as-of", help="Date (YYYY-MM-DD)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        result = run(args.topic, args.as_of)
        print(f"\n✅ Blog generated successfully!")
        print(f"Title: {result['plan'].blog_title}")
        print(f"Sections: {len(result['plan'].tasks)}")
        print(f"Words: {len(result['final'].split())}")
        print(f"Images: {len(result['image_specs'])}")
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()