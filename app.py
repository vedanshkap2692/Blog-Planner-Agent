"""
Streamlit Frontend for Blog Generator
======================================
Features:
- Seamless blog generation and display
- Past blogs browser with live preview
- Enhanced markdown rendering with local images
- Download options (MD, bundle, images)
- Real-time progress tracking
"""

from __future__ import annotations

import json
import os
import re
import zipfile
import time
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# Import backend
try:
    from backend import run as generate_blog, logger
except ImportError:
    st.error("❌ Failed to import backend. Ensure backend.py is in the same directory.")
    st.stop()

# ============================================================
# CONFIGURATION
# ============================================================

# Ensure images directory exists
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .blog-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
        transition: background-color 0.2s;
    }
    .blog-card:hover {
        background-color: #f5f5f5;
        cursor: pointer;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc66;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_slug(title: str) -> str:
    """Convert title to safe filename."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    """Create zip bundle with markdown and images."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(images_dir.parent))
                    z.write(p, arcname=arcname)
    
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    """Create zip of images only."""
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    
    files = [p for p in images_dir.rglob("*") if p.is_file()]
    if not files:
        return None
    
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arcname = str(p.relative_to(images_dir.parent))
            z.write(p, arcname=arcname)
    
    return buf.getvalue()


# ============================================================
# PAST BLOGS MANAGEMENT
# ============================================================

def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current directory, newest first.
    Excludes README and similar non-blog files.
    """
    cwd = Path(".")
    files = [
        p for p in cwd.glob("*.md") 
        if p.is_file() 
        and p.stem.lower() not in ["readme", "license", "changelog"]
        and not p.stem.startswith(".")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    """Read markdown file with error handling."""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to read {p}: {e}")
        return f"# Error\n\nFailed to read file: {e}"


def extract_title_from_md(md: str, fallback: str) -> str:
    """Extract title from first # heading."""
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def parse_metadata_from_md(md: str) -> Dict[str, Any]:
    """
    Extract metadata from markdown.
    Returns dict with title, word_count, images_count.
    """
    title = extract_title_from_md(md, "Untitled")
    word_count = len(md.split())
    
    # Count images
    img_pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
    images_count = len(img_pattern.findall(md))
    
    return {
        "title": title,
        "word_count": word_count,
        "images_count": images_count,
    }


# ============================================================
# MARKDOWN RENDERING
# ============================================================

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Optional[Path]:
    """Resolve image path (handles relative paths)."""
    src = src.strip().lstrip("./")
    
    # Try multiple path resolutions
    candidates = [
        Path(src),  # Direct path
        Path.cwd() / src,  # Current working directory
        Path(__file__).parent / src,  # App directory
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    
    return None


def render_markdown_with_local_images(md: str):
    """
    Render markdown with proper local image support.
    Handles captions and missing images gracefully.
    """
    matches = list(_MD_IMG_RE.finditer(md))
    
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    # Render parts
    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        # Handle image
        alt, src = payload.split("|||", 1)

        # Check for caption on next line
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        # Display image - FIXED: Remove use_column_width parameter
        if src.startswith("http://") or src.startswith("https://"):
            try:
                st.image(src, caption=caption or alt or None)
            except Exception as e:
                st.warning(f"⚠️ Failed to load remote image: {src}")
        else:
            img_path = _resolve_image_path(src)
            if img_path:
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=caption or alt or None)
                except Exception as e:
                    st.warning(f"⚠️ Failed to display image: `{src}` - {e}")
            else:
                st.warning(f"⚠️ Image not found: `{src}`")

        i += 1


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "selected_blog" not in st.session_state:
    st.session_state["selected_blog"] = None
if "generation_start_time" not in st.session_state:
    st.session_state["generation_start_time"] = None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("✍️ Blog Generator")
    
    # Generation section
    st.header("📝 Generate New Blog")
    
    topic = st.text_area(
        "Blog Topic",
        placeholder="Enter your blog topic or detailed description...",
        height=150,
        help="Be specific! Include target audience, key points, or angle."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        as_of = st.date_input("As-of date", value=date.today())
    with col2:
        log_level = st.selectbox(
            "Log Level",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0
        )
    
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)
    
    # Divider
    st.divider()
    
    # Past blogs section
    st.header("📚 Past Blogs")
    
    past_files = list_past_blogs()
    
    if not past_files:
        st.info("No saved blogs found.\n\nGenerate your first blog to get started!")
    else:
        st.caption(f"Found {len(past_files)} blog(s)")
        
        # Search/filter
        search = st.text_input(
            "🔍 Search blogs",
            placeholder="Filter by title...",
            label_visibility="collapsed"
        )
        
        # Build blog list
        displayed = 0
        for idx, p in enumerate(past_files[:30]):
            try:
                md_text = read_md_file(p)
                meta = parse_metadata_from_md(md_text)
                title = meta["title"]
                
                # Apply search filter
                if search and search.lower() not in title.lower():
                    continue
                
                displayed += 1
                
                # Modified time
                mtime = p.stat().st_mtime
                mod_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                
                # Create button
                if st.button(
                    f"**{title}**\n\n"
                    f"📅 {mod_date} · {meta['word_count']:,} words · "
                    f"{meta['images_count']} image(s)",
                    key=f"blog_{idx}",
                    use_container_width=True
                ):
                    st.session_state["selected_blog"] = p
                    st.session_state["last_out"] = {
                        "plan": type('Plan', (), {"blog_title": title})(),
                        "evidence": [],
                        "image_specs": [],
                        "final": md_text,
                    }
                    st.rerun()
            
            except Exception as e:
                logger.error(f"Error listing {p}: {e}")
                continue
        
        if search and displayed == 0:
            st.caption("No matching blogs found.")

# ============================================================
# MAIN CONTENT
# ============================================================

# Title
if st.session_state.get("selected_blog"):
    selected_file = st.session_state["selected_blog"]
    st.title(f"📖 {selected_file.stem.replace('_', ' ').title()}")
else:
    st.title("AI Blog Writer Agent")

# Tabs
tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    ["📝 Preview", "🧩 Plan", "🔎 Evidence", "🖼️ Images", "🧾 Logs"]
)

# ============================================================
# BLOG GENERATION
# ============================================================

if run_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a topic.")
        st.stop()
    
    # Clear previous selection
    st.session_state["selected_blog"] = None
    st.session_state["logs"] = []
    st.session_state["generation_start_time"] = time.time()
    
    # Create progress tracking
    status_container = st.empty()
    progress_bar = st.progress(0)
    log_container = st.expander("🔍 Live Progress", expanded=True)
    
    try:
        # Set log level
        import logging
        logger.setLevel(getattr(logging, log_level))
        
        # Phase 1: Router
        with log_container:
            st.text("🔍 Phase 1/5: Analyzing topic and routing...")
        status_container.info("📊 Phase 1/5: Analyzing topic...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Phase 2: Research
        with log_container:
            st.text("📚 Phase 2/5: Conducting research (if needed)...")
        status_container.info("📚 Phase 2/5: Researching...")
        progress_bar.progress(25)
        
        # Phase 3: Planning
        with log_container:
            st.text("🗺️ Phase 3/5: Creating blog outline...")
        status_container.info("🗺️ Phase 3/5: Planning structure...")
        progress_bar.progress(40)
        
        # Call backend (this is the main work)
        result = generate_blog(topic.strip(), as_of.isoformat())
        
        # Phase 4: Writing
        with log_container:
            st.text("✍️ Phase 4/5: Writing sections...")
        status_container.info("✍️ Phase 4/5: Writing content...")
        progress_bar.progress(70)
        time.sleep(0.5)
        
        # Phase 5: Images
        with log_container:
            st.text("🎨 Phase 5/5: Generating images...")
        status_container.info("🎨 Phase 5/5: Creating visuals...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # Complete
        progress_bar.progress(100)
        
        elapsed = time.time() - st.session_state["generation_start_time"]
        status_container.success(f"✅ Blog generation complete! (took {elapsed:.1f}s)")
        
        # Store result
        st.session_state["last_out"] = result
        
        # Log completion
        plan = result.get('plan')
        if plan:
            blog_title = plan.blog_title if hasattr(plan, 'blog_title') else "Blog"
            st.session_state["logs"].append(f"✅ Generated: {blog_title}")
            st.session_state["logs"].append(f"⏱️ Time: {elapsed:.1f}s")
            st.session_state["logs"].append(f"📝 Sections: {len(result.get('plan').tasks if hasattr(result.get('plan'), 'tasks') else [])}")
            st.session_state["logs"].append(f"🖼️ Images: {len(result.get('image_specs', []))}")
        
        # Wait to show completion
        time.sleep(1)
        
        st.rerun()
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        status_container.error(f"❌ Generation failed!")
        st.error(f"**Error:** {str(e)}")
        st.session_state["logs"].append(f"❌ ERROR: {e}")
        progress_bar.empty()

# ============================================================
# DISPLAY RESULTS
# ============================================================

out = st.session_state.get("last_out")

if out:
    final_md = out.get("final", "")
    plan_obj = out.get("plan")
    
    # Extract title
    if isinstance(plan_obj, dict):
        blog_title = plan_obj.get("blog_title", "Blog")
    elif plan_obj and hasattr(plan_obj, "blog_title"):
        blog_title = plan_obj.blog_title
    else:
        blog_title = extract_title_from_md(final_md, "Blog")
    
    # --- Preview Tab ---
    with tab_preview:
        if not final_md:
            st.warning("⚠️ No content available.")
        else:
            # Metadata metrics
            meta = parse_metadata_from_md(final_md)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Word Count", f"{meta['word_count']:,}")
            with col2:
                st.metric("🖼️ Images", meta['images_count'])
            with col3:
                sections = final_md.count("## ")
                st.metric("📑 Sections", sections)
            with col4:
                reading_time = max(1, meta['word_count'] // 200)
                st.metric("⏱️ Read Time", f"{reading_time} min")
            
            st.divider()
            
            # Render markdown
            render_markdown_with_local_images(final_md)
            
            # Download buttons
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            md_filename = f"{safe_slug(blog_title)}.md"
            
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                bundle = bundle_zip(final_md, md_filename, IMAGES_DIR)
                st.download_button(
                    "📦 Download Bundle (MD + Images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            with col3:
                # Copy to clipboard (markdown)
                if st.button("📋 Copy Markdown", use_container_width=True):
                    st.code(final_md, language="markdown")
                    st.success("Markdown displayed above - copy manually")
    
    # --- Plan Tab ---
    with tab_plan:
        st.subheader("📋 Blog Plan")
        
        if not plan_obj or not hasattr(plan_obj, 'tasks'):
            st.info("ℹ️ No plan data available (loaded from saved file).")
        else:
            # Convert plan to dict
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif hasattr(plan_obj, "__dict__"):
                plan_dict = plan_obj.__dict__
            else:
                plan_dict = {}
            
            # Metadata
            st.markdown(f"### {plan_dict.get('blog_title', 'Untitled')}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("👥 Audience", plan_dict.get("audience", "N/A"))
            col2.metric("🎭 Tone", plan_dict.get("tone", "N/A"))
            col3.metric("📚 Type", plan_dict.get("blog_kind", "N/A"))
            
            # Tasks
            tasks = plan_dict.get("tasks", [])
            if tasks:
                st.divider()
                st.subheader("📝 Section Breakdown")
                
                # Convert to DataFrame
                task_data = []
                for t in tasks:
                    task_data.append({
                        "ID": t.get("id"),
                        "Title": t.get("title"),
                        "Words": t.get("target_words"),
                        "Research": "✅" if t.get("requires_research") else "❌",
                        "Citations": "✅" if t.get("requires_citations") else "❌",
                        "Code": "✅" if t.get("requires_code") else "❌",
                    })
                
                df = pd.DataFrame(task_data).sort_values("ID")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Detailed breakdown
                with st.expander("🔍 View detailed task breakdown"):
                    for t in tasks:
                        st.markdown(f"#### {t.get('id')}. {t.get('title')}")
                        st.write(f"**Goal:** {t.get('goal')}")
                        st.write(f"**Target Words:** {t.get('target_words')}")
                        st.write("**Bullets:**")
                        for bullet in t.get("bullets", []):
                            st.write(f"- {bullet}")
                        
                        tags = t.get("tags", [])
                        if tags:
                            st.write(f"**Tags:** {', '.join(tags)}")
                        
                        st.divider()
    
    # --- Evidence Tab ---
    with tab_evidence:
        st.subheader("🔎 Research Evidence")
        
        evidence = out.get("evidence", [])
        
        if not evidence:
            st.info("ℹ️ No research evidence (closed-book mode or no results).")
        else:
            st.write(f"Found **{len(evidence)}** sources")
            
            # Convert to DataFrame
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e_dict = e.model_dump()
                elif hasattr(e, "__dict__"):
                    e_dict = e.__dict__
                else:
                    e_dict = e
                
                rows.append({
                    "Title": e_dict.get("title", ""),
                    "Source": e_dict.get("source", ""),
                    "Date": e_dict.get("published_at", ""),
                    "URL": e_dict.get("url", ""),
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Detailed view
            with st.expander("🔍 View full evidence details"):
                for idx, e in enumerate(evidence, 1):
                    if hasattr(e, "model_dump"):
                        e_dict = e.model_dump()
                    elif hasattr(e, "__dict__"):
                        e_dict = e.__dict__
                    else:
                        e_dict = e
                    
                    st.markdown(f"**{idx}. [{e_dict.get('title', 'Untitled')}]({e_dict.get('url', '#')})**")
                    st.write(f"*Source: {e_dict.get('source', 'Unknown')}*")
                    if e_dict.get('published_at'):
                        st.write(f"*Published: {e_dict.get('published_at')}*")
                    if e_dict.get('snippet'):
                        st.write(e_dict.get('snippet'))
                    st.divider()
    
    # --- Images Tab ---
    with tab_images:
        st.subheader("🖼️ Generated Images")
        
        specs = out.get("image_specs", [])
        
        if not specs and not IMAGES_DIR.exists():
            st.info("ℹ️ No images generated for this blog.")
        else:
            if specs:
                st.write(f"**Planned:** {len(specs)} image(s)")
                
                with st.expander("📋 View image specifications"):
                    for spec in specs:
                        st.json(spec)
                
                st.divider()
            
            if IMAGES_DIR.exists():
                files = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file()])
                
                if not files:
                    st.warning("⚠️ Images directory exists but is empty.")
                else:
                    st.write(f"**Saved:** {len(files)} file(s)")
                    
                    # Gallery view (2 columns)
                    cols = st.columns(2)
                    for idx, p in enumerate(files):
                        with cols[idx % 2]:
                            try:
                                img = Image.open(p)
                                st.image(img, caption=p.name, use_container_width=True)
                            except Exception as e:
                                st.error(f"Failed to load {p.name}: {e}")
                    
                    # Download zip
                    st.divider()
                    z = images_zip(IMAGES_DIR)
                    if z:
                        st.download_button(
                            "⬇️ Download All Images (ZIP)",
                            data=z,
                            file_name="blog_images.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
    
    # --- Logs Tab ---
    with tab_logs:
        st.subheader("🧾 Generation Logs")
        
        if not st.session_state["logs"]:
            st.info("ℹ️ No logs available.")
        else:
            log_text = "\n".join(st.session_state["logs"])
            
            st.text_area(
                "Event Log",
                value=log_text,
                height=400,
                disabled=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Logs",
                    data=log_text.encode("utf-8"),
                    file_name="blog_generation.log",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("🗑️ Clear Logs", use_container_width=True):
                    st.session_state["logs"] = []
                    st.rerun()

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to AI Blog Writer
    
    Generate high-quality technical blogs with AI assistance in minutes!
    
    ### ✨ Features
    
    - **🤖 Smart Research**: Automatic web research for up-to-date content
    - **📋 Structured Planning**: Organized outlines with word counts and goals
    - **✍️ Expert Writing**: Technical writing with proper terminology
    - **🎨 Image Generation**: AI-generated diagrams and illustrations
    - **📚 Citation Support**: Proper source attribution and references
    - **📦 Export Options**: Download as Markdown or complete bundle
    
    ### 🚀 Getting Started
    
    1. **Enter Topic**: Describe your blog topic in the sidebar
    2. **Set Date**: Choose the as-of date for research
    3. **Generate**: Click "Generate Blog" and wait for the magic ✨
    4. **Review**: Browse through tabs to review plan, evidence, and content
    5. **Download**: Export your blog in various formats
    
    ### 📚 Past Blogs
    
    Access your previously generated blogs from the sidebar. Click any blog to load and view it instantly.
    
    ### 💡 Tips for Best Results
    
    - Be specific with your topic (include target audience, key points, angle)
    - Use recent dates for trending topics
    - Review the plan before downloading
    - Check citations for accuracy
    
    ---
    
    **Ready to create amazing content?** Enter your topic in the sidebar and click Generate! 🚀
    """)
    
    # Show sample topics
    with st.expander("💡 Sample Topics to Try"):
        st.markdown("""
        **Technical Tutorials:**
        - "Building a REST API with FastAPI and PostgreSQL for beginners"
        - "Complete guide to Docker containerization for Python applications"
        
        **Explainers:**
        - "How Does Kubernetes Handle Container Orchestration?"
        - "Understanding OAuth 2.0: A comprehensive guide for developers"
        
        **Comparisons:**
        - "Redis vs Memcached: Which caching solution is right for your project?"
        - "GraphQL vs REST: Choosing the right API architecture in 2026"
        
        **System Design:**
        - "Designing a scalable URL shortener service like bit.ly"
        - "Building a real-time chat application: Architecture and best practices"
        
        **News Roundup:**
        - "Latest updates in Python 3.13: What developers need to know"
        - "AI and Machine Learning trends in 2026: A technical overview"
        """)
    
    # Statistics (if any blogs exist)
    past_files = list_past_blogs()
    if past_files:
        st.divider()
        st.subheader("📊 Your Statistics")
        
        total_blogs = len(past_files)
        total_words = 0
        total_images = 0
        
        for p in past_files[:50]:  # Limit to 50 for performance
            try:
                md = read_md_file(p)
                meta = parse_metadata_from_md(md)
                total_words += meta['word_count']
                total_images += meta['images_count']
            except:
                pass
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📝 Total Blogs", total_blogs)
        col2.metric("📊 Total Words", f"{total_words:,}")
        col3.metric("🖼️ Total Images", total_images)