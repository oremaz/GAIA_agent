import asyncio
import csv
import io
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import torch
from ddgs import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from custom_models import get_or_create_qwen3_gguf_embedding, get_or_create_jina_reranker


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def safe_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None


safe_globals = {
    "__builtins__": {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "sum": sum,
        "max": max,
        "min": min,
        "round": round,
        "abs": abs,
        "sorted": sorted,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,
        "type": type,
        "isinstance": isinstance,
        "print": print,
        "open": open,
        "bool": bool,
        "set": set,
        "tuple": tuple,
    }
}

core_modules = [
    "math",
    "datetime",
    "re",
    "os",
    "sys",
    "json",
    "csv",
    "random",
    "itertools",
    "collections",
    "functools",
    "operator",
    "copy",
    "decimal",
    "fractions",
    "uuid",
    "typing",
    "statistics",
    "pathlib",
    "glob",
    "shutil",
    "tempfile",
    "pickle",
    "gzip",
    "zipfile",
    "tarfile",
    "base64",
    "hashlib",
    "secrets",
    "hmac",
    "textwrap",
    "string",
    "difflib",
    "socket",
    "ipaddress",
    "logging",
    "warnings",
    "traceback",
    "pprint",
    "threading",
    "queue",
    "sqlite3",
    "urllib",
    "html",
    "xml",
    "configparser",
]

for module in core_modules:
    imported = safe_import(module)
    if imported:
        safe_globals[module] = imported

optional_modules = {
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "pd": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "plt": "matplotlib.pyplot",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "plotly": "plotly",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "PIL": "PIL",
    "skimage": "skimage",
    "pytz": "pytz",
    "requests": "requests",
    "bs4": "bs4",
    "sympy": "sympy",
    "tqdm": "tqdm",
    "yaml": "yaml",
    "toml": "toml",
}

for alias, module_name in optional_modules.items():
    imported = safe_import(module_name)
    if imported:
        safe_globals[alias] = imported

if safe_globals.get("bs4"):
    safe_globals["BeautifulSoup"] = safe_globals["bs4"].BeautifulSoup

if safe_globals.get("PIL"):
    image_module = safe_import("PIL.Image")
    if image_module:
        safe_globals["Image"] = image_module


def execute_python_code(code: str) -> str:
    code = code.strip()
    if not code:
        return "No code provided."

    local_vars: Dict[str, object] = {}

    try:
        compiled = compile(code, "<assistant>", "eval")
        result = eval(compiled, safe_globals, local_vars)
        return f"{result}"
    except SyntaxError:
        try:
            compiled = compile(code, "<assistant>", "exec")
            exec(compiled, safe_globals, local_vars)
            if "_result" in local_vars:
                return f"{local_vars['_result']}"
            return "Code executed successfully."
        except Exception as exc:
            return f"Error executing code: {exc}"
    except Exception as exc:
        return f"Error executing code: {exc}"


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
    return chunks


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b) + 1e-12)


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {".txt", ".md", ".py", ".log"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext in {".json"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        if ext in {".csv", ".tsv"}:
            delimiter = "," if ext == ".csv" else "\t"
            with open(file_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = ["\t".join(row) for row in reader]
            return "\n".join(rows)
        if ext in {".pdf"}:
            try:
                import fitz

                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            except Exception:
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(file_path)
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as exc:
                    return f"Could not parse PDF: {exc}"
        if ext in {".docx"}:
            try:
                import docx

                doc = docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception as exc:
                return f"Could not parse DOCX: {exc}"

        with open(file_path, "rb") as f:
            sample = f.read(4096).decode("utf-8", errors="ignore")
        return sample
    except Exception as exc:
        return f"Failed to read {file_path}: {exc}"


@dataclass
class KnowledgeChunk:
    text: str
    source: str
    embedding: List[float]


class SimpleKnowledgeBase:
    def __init__(self, max_chunks: int = 400):
        self.embedder = get_or_create_qwen3_gguf_embedding()
        self.reranker = get_or_create_jina_reranker(
            model_name="jinaai/jina-reranker-v3", device="cpu", top_n=5
        )
        self.max_chunks = max_chunks
        self.chunks: List[KnowledgeChunk] = []
        self.sources: List[str] = []
        self._lock = threading.Lock()

    def add_text(self, text: str, source: str) -> int:
        text = (text or "").strip()
        if not text:
            return 0
        pieces = chunk_text(text)
        added = 0
        with self._lock:
            for piece in pieces:
                try:
                    emb = self.embedder._get_text_embedding(piece)
                except Exception as exc:
                    logger.warning("Embedding failed for chunk: %s", exc)
                    continue
                self.chunks.append(KnowledgeChunk(text=piece, source=source, embedding=emb))
                added += 1
            if source not in self.sources:
                self.sources.append(source)
            if len(self.chunks) > self.max_chunks:
                self.chunks = self.chunks[-self.max_chunks :]
        return added

    def add_file(self, file_path: str) -> bool:
        text = extract_text_from_file(file_path)
        if not text:
            return False
        added = self.add_text(text, source=os.path.abspath(file_path))
        return added > 0

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, KnowledgeChunk]]:
        with self._lock:
            if not self.chunks:
                return []
            try:
                query_vec = self.embedder._get_query_embedding(query)
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)
                return []

            scored = [
                (cosine_similarity(query_vec, chunk.embedding), chunk)
                for chunk in self.chunks
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            prelim = [chunk for _, chunk in scored[: max(top_k * 2, 6)]]
            if not prelim:
                return []
            try:
                pairs = [[query, chunk.text] for chunk in prelim]
                rerank_scores = self.reranker.score_text_pairs(pairs)
                zipped = list(zip(rerank_scores, prelim))
                zipped.sort(key=lambda x: x[0], reverse=True)
                return zipped[:top_k]
            except Exception as exc:
                logger.warning("Reranker failed: %s", exc)
                return scored[:top_k]

    def format_retrieval(self, query: str, top_k: int = 5) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return "Knowledge base empty."
        lines = []
        for idx, (score, chunk) in enumerate(hits, start=1):
            snippet = chunk.text.strip().replace("\n", " ")
            snippet = snippet[:600]
            lines.append(f"[DOC {idx} | score={score:.3f} | {chunk.source}] {snippet}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "chunks": len(self.chunks),
                "sources": list(self.sources),
            }


class SimpleWebFetcher:
    def __init__(self, knowledge_base: SimpleKnowledgeBase):
        self.knowledge_base = knowledge_base

    def search_and_ingest(self, query: str) -> str:
        url = self._search(query)
        if not url:
            return "Search failed. Try a different query."
        text = self._fetch_text(url)
        if not text or "Error" in text:
            return f"Failed to extract content from {url}: {text}"
        added = self.knowledge_base.add_text(text, source=url)
        return f"Added {added} chunks from {url}" if added else f"No text extracted from {url}"

    def _search(self, query: str) -> Optional[str]:
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=1, backend="google"))
            if results:
                return results[0].get("href") or results[0].get("url")
        except Exception as exc:
            logger.warning("DDGS search failed: %s", exc)
        return None

    def _fetch_text(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=20, headers=headers)
            resp.raise_for_status()
            if BeautifulSoup:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                return soup.get_text(" ", strip=True)
            cleaned = re.sub(r"<[^>]+>", " ", resp.text)
            return re.sub(r"\s+", " ", cleaned)
        except Exception as exc:
            return f"Error fetching {url}: {exc}"


class GptOssReasoner:
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._tokenizer = None
        self._model = None
        self._tool_registry: Dict[str, Dict[str, object]] = {}
        self.tool_schemas: List[dict] = []
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            self._load()

    def _load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

    def add_tool(self, name: str, func, description: str):
        if name in self._tool_registry:
            return
        self._tool_registry[name] = {"func": func, "description": description}
        self.tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            }
        )

    TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*([A-Za-z0-9_]+)\("([\s\S]*?)"\)')
    JSON_TOOL_CALL_RE = re.compile(r'\{"query":\s*"([^"]+)"(?:,\s*"[^"]*":\s*[^}]*)?\}')

    def _extract_calls(self, text: str):
        calls = [(m.group(1), m.group(2)) for m in self.TOOL_CALL_RE.finditer(text)]
        if not calls:
            for match in self.JSON_TOOL_CALL_RE.finditer(text):
                calls.append(("enhanced_web_search", match.group(1)))
        return calls

    def _exec_calls(self, calls):
        outputs = []
        for name, arg in calls:
            tool = self._tool_registry.get(name)
            if not tool:
                outputs.append(f"Tool {name} not found")
                continue
            try:
                outputs.append(str(tool["func"](arg)))
            except Exception as exc:
                outputs.append(f"Tool {name} error: {exc}")
        return outputs

    def solve(self, question: str, max_iterations: int = 5, reasoning_effort: str = "medium") -> str:
        self._ensure_model()
        is_gpt = "gpt-oss" in self.model_name.lower()
        if not is_gpt or not self._tool_registry:
            return self._solve_simple(question, reasoning_effort)
        return self._solve_with_tools(question, max_iterations, reasoning_effort)

    def _solve_simple(self, question: str, reasoning_effort: str) -> str:
        messages = [{"role": "user", "content": question}]
        tokenizer = self._tokenizer
        assert tokenizer is not None
        model = self._model
        assert model is not None
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        response = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return response.strip()

    def _solve_with_tools(self, question: str, max_iterations: int, reasoning_effort: str) -> str:
        history: List[str] = []
        current_prompt = question
        instructions = (
            "You are a GAIA competition assistant. Call tools using TOOL_CALL: name(\"input\") "
            "to gather facts or computations. When you are certain, answer with FINAL ANSWER: <answer>."
        )
        for iteration in range(max_iterations):
            composed = f"{instructions}\nQUESTION: {question}\nCURRENT PROMPT:\n{current_prompt}"
            response = self._solve_simple(composed, reasoning_effort if iteration == 0 else "medium")
            history.append(response)
            if "FINAL ANSWER:" in response:
                return response
            calls = self._extract_calls(response)
            if not calls:
                current_prompt = (
                    "If you have enough information, output FINAL ANSWER: <answer>. "
                    "Otherwise issue a TOOL_CALL using one of the available tools."
                )
                continue
            results = self._exec_calls(calls)
            tool_context = "\n".join(
                f"TOOL_RESULT {i+1} ({calls[i][0]}): {result}" for i, result in enumerate(results)
            )
            current_prompt = (
                f"Tool results obtained:\n{tool_context}\nUse them to progress and then provide FINAL ANSWER when ready."
            )
        return history[-1] if history else "Max iterations reached without FINAL ANSWER."


class TextOnlyGptOssAgent:
    def __init__(self):
        self.knowledge_base = SimpleKnowledgeBase()
        self.reasoner = GptOssReasoner()
        self.web_fetcher = SimpleWebFetcher(self.knowledge_base)
        self.reasoner.add_tool(
            "enhanced_web_search",
            self._web_search_tool,
            "Search the web and ingest the first result into the knowledge base.",
        )
        self.reasoner.add_tool(
            "dynamic_knowledge_query",
            self._knowledge_query_tool,
            "Retrieve the most relevant passages from the local knowledge base.",
        )
        self.reasoner.add_tool(
            "execute_python_code",
            execute_python_code,
            "Execute Python code safely and return the output.",
        )

    def _web_search_tool(self, query: str) -> str:
        return self.web_fetcher.search_and_ingest(query)

    def _knowledge_query_tool(self, query: str) -> str:
        return self.knowledge_base.format_retrieval(query, top_k=4)

    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> Optional[str]:
        if not task_id:
            return None
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()
            content_disp = response.headers.get("content-disposition", "")
            match = re.search(r'filename="(.+)"', content_disp)
            filename = match.group(1) if match else f"{task_id}.bin"
            with open(filename, "wb") as f:
                f.write(response.content)
            return os.path.abspath(filename)
        except Exception as exc:
            logger.exception("Failed to download GAIA file %s: %s", task_id, exc)
            return None

    def add_documents_to_knowledge_base(self, file_path: str):
        if not file_path:
            return False
        return self.knowledge_base.add_file(file_path)

    def get_knowledge_base_stats(self):
        return self.knowledge_base.stats()

    async def solve_gaia_question(self, question_data: Dict[str, object]) -> str:
        question = str(question_data.get("Question", ""))
        task_id = str(question_data.get("task_id", ""))
        file_path = None
        if task_id:
            file_path = self.download_gaia_file(task_id)
            if file_path:
                self.add_documents_to_knowledge_base(file_path)
        if not question:
            return "No question provided."

        context_preview = self.knowledge_base.format_retrieval(question, top_k=3)
        gaia_rules = (
            "YOUR FINAL ANSWER should be as short as possible. "
            "If asked for numbers, do not include units unless required. "
            "For strings, do not include articles. "
            "For comma separated lists, follow the rules per element."
        )
        prompt = (
            f"GAIA Task ID: {task_id or 'N/A'}\n"
            f"Question: {question}\n"
            f"Current knowledge base preview:\n{context_preview}\n\n"
            "You can call tools such as enhanced_web_search, dynamic_knowledge_query, and execute_python_code "
            "using TOOL_CALL syntax when needed. Reason carefully and provide FINAL ANSWER once certain.\n"
            f"{gaia_rules}"
        )
        return self.reasoner.solve(prompt, max_iterations=6, reasoning_effort="high")


async def _demo():
    agent = TextOnlyGptOssAgent()
    question = {"Question": "Who won the 2012 Nobel prize in Literature?", "task_id": ""}
    answer = await agent.solve_gaia_question(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(_demo())
