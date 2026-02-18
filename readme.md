# Eve AI Blog Generator

## Overview

**Eve AI** is a production-grade, fully automated technical blog generator. It leverages advanced LLMs (via OpenRouter), LangGraph for workflow orchestration, and Gemini image models for diagram generation. The system is engineered for reliability, extensibility, and high-quality technical content creation with minimal human intervention.

---

## Features

- **Automated Blog Generation**: From a single topic prompt, Eve AI produces a multi-section, well-structured technical blog post.
- **Research Integration**: Optionally performs web research using Tavily, deduplicates sources, and grounds content in up-to-date evidence.
- **Section Planning**: Uses LLMs to create actionable outlines with goals, bullets, and word targets for each section.
- **Parallelized Writing**: Each section is written independently, allowing for scalable and fast content generation.
- **Image Planning & Generation**: Decides where technical diagrams are needed, generates prompts, and uses Gemini via OpenRouter to create and insert images.
- **LangGraph Orchestration**: All steps are managed as a robust, stateful graph, enabling modularity, error handling, and extensibility.
- **Rate Limiting & Concurrency Control**: Built-in mechanisms to avoid API abuse and ensure smooth operation.
- **Comprehensive Logging**: Rotating logs for debugging and auditability.
- **CLI & API Ready**: Usable as a command-line tool or as a Python module.

---

## Architecture

### 1. **LangGraph State Machine**

The core workflow is modeled as a directed acyclic graph (DAG) using [LangGraph](https://github.com/langchain-ai/langgraph):

```
START
  |
[router] --(needs_research?)--> [research]
  |                                 |
  +-----------------------------+   |
                                |   v
                            [orchestrator]
                                |
                             [fanout]
                                |
                             [worker] (parallel for each section)
                                |
                            [reducer subgraph]
                                |
                               END
```

#### **Reducer Subgraph** (for post-processing):

```
[merge_content] -> [decide_images] -> [generate_and_place_images]
```

- **merge_content**: Merges all section markdown into a single document.
- **decide_images**: Uses an LLM to decide where images/diagrams are needed and generates prompts/placeholders.
- **generate_and_place_images**: Calls OpenRouter Gemini to generate images, saves them, and replaces placeholders with markdown image links.

### 2. **Nodes & Their Roles**

- **router_node**: Decides if research is needed and generates search queries.
- **research_node**: Uses Tavily to fetch and deduplicate relevant web evidence.
- **orchestrator_node**: Plans the blog structure (sections, goals, bullets, word counts).
- **fanout**: Splits the plan into independent section-writing tasks.
- **worker_node**: Writes each section using the plan and evidence.
- **merge_content**: Merges all sections into a single markdown document.
- **decide_images**: Determines where images are needed and what they should depict.
- **generate_and_place_images**: Generates images using OpenRouter Gemini and inserts them into the markdown.

---

## Engineering Details

### **1. Rate Limiting & Concurrency**

- Uses a custom `RateLimiter` class with a semaphore and minimum interval between API calls.
- Prevents API overload and ensures compliance with OpenRouter and Tavily rate limits.

### **2. Error Handling**

- All nodes are wrapped with try/except and log errors with stack traces.
- Graceful fallback for image generation: if an image fails, a markdown block with the error and prompt is inserted.

### **3. Logging**

- Rotating file and console logging via Python's `logging` and `RotatingFileHandler`.
- Logs all major events, errors, and API interactions.

### **4. Configuration**

- All secrets and API keys are loaded from `.env` (never committed to git).
- Model names, limits, and timeouts are configurable via the `Config` class.

### **5. Image Generation**

- Uses OpenRouter's Gemini image models (`google/gemini-3-pro-image-preview` or `google/gemini-2.5-flash-image-preview`).
- Handles all OpenRouter response formats (base64, URL, nested structures).
- Images are saved to the `images/` directory and referenced in the markdown.

### **6. Schema Validation**

- All data structures (tasks, plans, evidence, images) are validated with Pydantic models.
- Ensures strict contract between LLM outputs and downstream processing.

### **7. CLI Usage**

- Run from the command line:
  ```sh
  python backend.py "Your blog topic here"
  ```
- Options for log level and date override.

---

## File Structure

```
eve_AI/
├── backend.py         # Main backend and CLI
├── app.py             # Streamlit or web UI (optional)
├── .env               # API keys (not committed)
├── .gitignore         # Excludes .env, venv, __pycache__, images, logs, etc.
├── images/            # Generated images
├── logs/              # Log files
├── experiments/       # Notebooks and research
└── README.md          # This file
```

---

## How to Use

1. **Clone the repo and install dependencies**  
   ```sh
   git clone https://github.com/yourusername/eve_AI.git
   cd eve_AI
   pip install -r requirements.txt
   ```

2. **Set up your `.env` file**  
   ```
   OPEN_ROUTER_API_KEY=your-openrouter-key
   TAVILY_API_KEY=your-tavily-key
   ```

3. **Run the generator**  
   ```sh
   python backend.py "How async tools impact Python development"
   ```

4. **Output**  
   - Markdown blog file in the project root.
   - Images in the `images/` directory.
   - Logs in `blog_generator.log`.

---

## Extending & Customizing

- **Add new nodes**: Just define a function and add it to the LangGraph.
- **Change models**: Update the `Config` class.
- **Tune prompts**: Edit the system prompt strings for each node.
- **Swap image model**: Change `config.image_model` to any OpenRouter-supported image model.

---

## Security & Best Practices

- **Never commit `.env` or API keys.**
- **All generated content and logs are excluded from git via `.gitignore`.**
- **Handles API errors and rate limits gracefully.**

---

## Dependencies

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenRouter](https://openrouter.ai/)
- [Tavily](https://tavily.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)
- [Requests](https://docs.python-requests.org/)
- [Streamlit](https://streamlit.io/) (optional, for UI)

---

## Example Output

- **Markdown blog**: `mastering_async_tools_in_python_a_complete_tutorial_and_their_impact_on_modern_development.md`
- **Images**:  
  - `images/async_sync_execution_comparison.png`
  - `images/asyncio_performance_security.png`
  - `images/async_tools_impact_python_ecosystem.png`

---

## License

MIT License

---

## Authors

- [Your Name](https://github.com/yourusername)
- Contributions welcome!

---

## Acknowledgements

- Inspired by the LangGraph and OpenRouter communities.
- Thanks to all open-source contributors.
