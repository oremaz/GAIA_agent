import os
import requests
import base64
import re
import mimetypes
import logging
import sys
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from smolagents import CodeAgent, OpenAIServerModel, Tool, tool

# Langfuse observability imports
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

from markdownify import markdownify
from requests.exceptions import RequestException


from google import genai
from google.genai import types
from openai import OpenAI
from ddgs import DDGS

_GENAI_CLIENT = None
_OPENAI_CLIENT = None
_FORMAT_PROVIDER = None
_FORMAT_MODEL_NAME = None
logger = logging.getLogger(__name__)

# Send logs to stdout so they show up in notebooks/terminals.
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    sh.setLevel(logging.INFO)
    root_logger.addHandler(sh)
root_logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

def llm_reformat(response: str, question: str) -> str:
    """Extract the final answer from a response using an LLM.

    Args:
        response: Full model response text.
        question: Original user question.

    Returns:
        Extracted final answer, or the original response on failure.
    """
    format_prompt = f"""Extract the exact answer from the response below.

Now extract the exact answer:
Question: {question}
Response: {response}

Provide your reasoning, then the exact answer."""

    try:
        def _extract_final(text: str) -> str:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                return text.strip()
            last = lines[-1]
            for prefix in ("Answer:", "Final answer:", "FINAL ANSWER:"):
                if last.startswith(prefix):
                    last = last[len(prefix):].strip()
            return last

        provider = _FORMAT_PROVIDER or "gemini"
        model_name = _FORMAT_MODEL_NAME or "gemini-3-pro-preview"

        if provider == "openai":
            global _OPENAI_CLIENT
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return response
            if _OPENAI_CLIENT is None:
                _OPENAI_CLIENT = OpenAI(api_key=api_key)

            completion = _OPENAI_CLIENT.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": format_prompt}],
            )
            text = (completion.choices[0].message.content or "").strip()
            return _extract_final(text) if text else response

        global _GENAI_CLIENT
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return response
        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        formatting_response = _GENAI_CLIENT.models.generate_content(
            model=model_name,
            contents=[format_prompt]
        )
        text = (formatting_response.text or "").strip()
        return _extract_final(text) if text else response
    except Exception as e:
        logger.warning("LLM reformatting failed: %s", e)
        return response
    
class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem with optional formatting."
    inputs = {
        "answer": {
            "type": "any", 
            "description": "The final answer to the problem"
        },
        "original_question": {
            "type": "string", 
            "description": "The original question to determine answer format", 
            "nullable": True  # Optional parameter
        }
    }
    output_type = "any"

    def forward(self, answer: Any, original_question: str = "") -> Any:
        """Return the answer, optionally normalized by the LLM.

        Args:
            answer: Proposed final answer.
            original_question: Optional original question for formatting.

        Returns:
            Normalized answer if possible, otherwise the original answer.
        """
        if not original_question:
            return answer
        if isinstance(answer, (int, float)):
            return answer
        if isinstance(answer, str):
            return llm_reformat(answer, original_question)
        return answer

@tool
def get_youtube_transcript(youtube_url: str) -> str:
    """Fetch a YouTube transcript given a URL or video id.

    Args:
        youtube_url: YouTube URL or video id.

    Returns:
        Transcript text if available, otherwise an error message.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

        video_id = None
        if "youtube" in youtube_url or "youtu.be" in youtube_url:
            match = re.search(r"(?:v=|youtu.be/)([\w-]{11})", youtube_url)
            if match:
                video_id = match.group(1)
        else:
            if re.fullmatch(r"[\w-]{11}", youtube_url.strip()):
                video_id = youtube_url.strip()

        if not video_id:
            return "Could not extract a valid YouTube video ID."

        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
        return transcript_text
    except ImportError:
        return "Install 'youtube-transcript-api' to use this tool: pip install youtube-transcript-api."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

class WebSearchTool(Tool):
    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.ddgs = DDGS(**kwargs)

    def _perform_search(self, query: str):
        """Perform the underlying DDGS search.

        Args:
            query: Search query.

        Returns:
            Iterable of DDGS search results.
        """
        return self.ddgs.text(query, max_results=self.max_results)

    def forward(self, query: str) -> str:
        results = []
        
        # First attempt with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self._perform_search, query)
                results = future.result(timeout=30)  # 30 second timeout
            except TimeoutError:
                logger.warning("First search attempt timed out after 30 seconds, retrying...")
                results = []
        
        # Retry if no results or timeout occurred
        if len(results) == 0:
            logger.info("Retrying search...")
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(self._perform_search, query)
                    results = future.result(timeout=30)  # 30 second timeout for retry
                except TimeoutError:
                    raise Exception("Search timed out after 30 seconds on both attempts. Try a different query.")
        
        # Final check for results
        if len(results) == 0:
            raise Exception("No results found after two attempts! Try a less restrictive/shorter query.")
        
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

@tool
def visit_webpage(url: str) -> str:
    """Fetch a webpage and return its content as Markdown.

    Args:
        url: Webpage URL.

    Returns:
        Markdown content, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the content as HTML with BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text and convert to Markdown
        content = soup.get_text(separator="\n", strip=True)
        markdown_content = markdownify(content)
        # Clean up the markdown content
        markdown_content = re.sub(r'\n+', '\n', markdown_content)  # Remove excessive newlines
        markdown_content = re.sub(r'\s+', ' ', markdown_content)  # Remove excessive spaces
        markdown_content = markdown_content.strip()  # Strip leading/trailing whitespace
        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

class UnifiedMultimodalTool(Tool):
    name = "multimodal_processor"
    description = """
    Unified tool for processing audio, video, and image files.
    Supports transcription, analysis, content extraction, captioning, and cross-modal tasks.
    Handles common formats: mp3, wav, m4a, mp4, avi, mov, jpg, png, gif, etc.
    OpenAI mode uses Responses API for images and the Audio Transcriptions endpoint for audio/video.
    NOTE: PDFs are NOT supported by this tool (use Docling for PDF processing).
    """
    
    # Required: Define inputs attribute
    inputs = {
        "file_path": {
            "type": "string", 
            "description": "Path to the media file (audio, video, or image) to process"
        },
        "task": {
            "type": "string",
            "description": "Processing task: analyze, transcribe, extract, caption, summarize, or search",
            "nullable": True  # Optional parameter
        },
        "modality": {
            "type": "string", 
            "description": "Force specific modality: auto, audio, video, or image",
            "nullable": True  # Optional parameter
        },
        "additional_context": {
            "type": "string",
            "description": "Extra instructions for processing",
            "nullable": True  # Optional parameter
        }
    }
    
    # Required: Define output type
    output_type = "string"
    
    def __init__(self, provider: str, model_name: str, api_key: str):
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.transcription_model = self._select_transcription_model()

        if provider == "gemini":
            self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider for multimodal tool: {provider}")
        # Initialize mimetypes for better detection
        mimetypes.init()
        
    def forward(self, 
                file_path: str, 
                task: str = "analyze", 
                modality: str = "auto",
                additional_context: str = "") -> str:
        """Process multimedia files with a unified interface.

        Args:
            file_path: Path to media file.
            task: Processing task (analyze, transcribe, extract, caption, summarize, search).
            modality: Force modality (auto, audio, video, image).
            additional_context: Extra instructions for processing.

        Returns:
            Model output text or an error message.
        """
        try:
            # Auto-detect modality if not specified
            if modality == "auto":
                modality = self._detect_modality(file_path)
            
            # Reject PDFs - they should use Docling
            if modality in ['pdf', 'unsupported']:
                return f"Error: PDF files are not supported by this tool. Please use Docling for PDF processing."
            
            # Generate appropriate prompt based on task and modality
            prompt = self._generate_prompt(task, modality, additional_context)

            if self.provider == "openai":
                if modality == "image":
                    return self._process_openai_image(file_path, prompt)
                if modality in {"audio", "video"}:
                    return self._process_openai_audio(file_path)
                return f"Unsupported modality for OpenAI: {modality}"

            # Gemini flow
            file_size = os.path.getsize(file_path)
            use_files_api = file_size > 20 * 1024 * 1024  # 20MB threshold

            if use_files_api:
                return self._process_with_files_api(file_path, prompt, modality)
            return self._process_inline(file_path, prompt, modality)
                
        except Exception as e:
            return f"Error processing {modality} file: {str(e)}"
    
    def _detect_modality(self, file_path: str) -> str:
        """Detect media modality using mimetypes and extension fallback.

        Args:
            file_path: Path to media file.

        Returns:
            Detected modality string (audio, video, image, unsupported, unknown).
        """
        # Use mimetypes.guess_type for reliable MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type:
            if mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type.startswith('video/'):
                return 'video'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type == 'application/pdf':
                # PDFs are not supported - should use Docling
                return 'unsupported'
        
        # Fallback to extension-based detection if MIME type is unknown
        ext = file_path.lower().split('.')[-1]
        audio_exts = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'wma'}
        video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', '3gp'}
        image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'svg'}
        
        if ext in audio_exts:
            return 'audio'
        elif ext in video_exts:
            return 'video'
        elif ext in image_exts:
            return 'image'
        elif ext == 'pdf':
            # PDFs are not supported - should use Docling
            return 'unsupported'
        else:
            return 'unknown'
    
    def _get_mime_type(self, file_path: str) -> str:
        """Determine MIME type with fallback mappings.

        Args:
            file_path: Path to media file.

        Returns:
            MIME type string.
        """
        mime_type, encoding = mimetypes.guess_type(file_path)
        
        if mime_type:
            return mime_type
        
        # Fallback mappings for common formats not detected by mimetypes
        ext = file_path.lower().split('.')[-1]
        fallback_mappings = {
            'm4a': 'audio/mp4',
            'mkv': 'video/x-matroska',
            'webm': 'video/webm',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            'webp': 'image/webp',
            'pdf': 'application/pdf'
        }
        
        return fallback_mappings.get(ext, 'application/octet-stream')
    
    def _generate_prompt(self, task: str, modality: str, context: str) -> str:
        """Generate a task prompt based on modality and context.

        Args:
            task: Task name.
            modality: Media modality.
            context: Additional instructions.

        Returns:
            Prompt string tailored to the task and modality.
        """
        
        base_prompts = {
            'analyze': {
                'audio': 'Analyze this audio file. Describe the content, identify sounds, speech, music, and any notable audio characteristics.',
                'video': 'Analyze this video comprehensively. Describe visual content, actions, audio elements, and their relationship.',
                'image': 'Analyze this image in detail. Describe objects, scenes, text, colors, composition, and any notable features.',
            },
            'transcribe': {
                'audio': 'Transcribe all speech in this audio file accurately. Include speaker changes if multiple speakers.',
                'video': 'Transcribe all speech and dialogue in this video. Note visual context when relevant.',
                'image': 'Extract and transcribe any text visible in this image using OCR.',
            },
            'extract': {
                'audio': 'Extract key information, topics, or specific content from this audio.',
                'video': 'Extract key visual and audio information from this video.',
                'image': 'Extract all text, objects, and important visual elements from this image.',
            },
            'caption': {
                'audio': 'Generate descriptive captions for this audio content.',
                'video': 'Generate detailed captions describing both visual and audio elements of this video.',
                'image': 'Generate a comprehensive caption describing this image.',
            },
            'summarize': {
                'audio': 'Provide a concise summary of the main points in this audio.',
                'video': 'Summarize the key visual and audio content of this video.',
                'image': 'Summarize the main elements and content of this image.',
            },
            'search': {
                'audio': 'Make this audio content searchable by extracting keywords, topics, and semantic information.',
                'video': 'Extract searchable content from both visual and audio elements of this video.',
                'image': 'Extract searchable keywords and descriptions from this image.',
            }
        }
        
        prompt = base_prompts.get(task, base_prompts['analyze']).get(modality, 'Analyze this media file.')
        
        if context:
            prompt += f" Additional context: {context}"
            
        return prompt
    
    def _get_media_resolution(self, modality: str) -> Optional[Dict[str, str]]:
        """Return recommended media resolution hint for Gemini.

        Args:
            modality: Media modality.

        Returns:
            Resolution hint dictionary or None.
        """
        if modality == 'image':
            return {"level": "media_resolution_high"}
        elif modality == 'video':
            return {"level": "media_resolution_low"}
        # PDFs are not supported by this tool
        return None

    def _process_with_files_api(self, file_path: str, prompt: str, modality: str) -> str:
        """Process large files using the Gemini Files API.

        Args:
            file_path: Path to media file.
            prompt: Prepared prompt text.
            modality: Media modality.

        Returns:
            Model response text.
        """
        uploaded_file = self.client.files.upload(file=file_path)
        
        resolution = self._get_media_resolution(modality)
        
        # Create content with media resolution if applicable
        if resolution:
            contents = [
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type
                            ),
                            media_resolution=resolution
                        )
                    ]
                )
            ]
        else:
            contents = [prompt, uploaded_file]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        return response.text
    
    def _process_inline(self, file_path: str, prompt: str, modality: str) -> str:
        """Process smaller files inline with the Gemini API.

        Args:
            file_path: Path to media file.
            prompt: Prepared prompt text.
            modality: Media modality.

        Returns:
            Model response text.
        """
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        mime_type = self._get_mime_type(file_path)
        resolution = self._get_media_resolution(modality)
        
        # Correct way to create parts in the new SDK with media resolution
        if resolution:
            contents = [
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                data=file_data,
                                mime_type=mime_type
                            ),
                            media_resolution=resolution
                        )
                    ]
                )
            ]
        else:
            contents = [
                prompt,
                types.Part.from_bytes(
                    data=file_data,
                    mime_type=mime_type
                )
            ]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

        return response.text

    def _select_transcription_model(self) -> str:
        """Choose an OpenAI transcription model.

        Returns:
            Transcription model name.
        """
        env_model = os.environ.get("OPENAI_TRANSCRIBE_MODEL")
        if env_model:
            return env_model
        return "gpt-4o-mini-transcribe"

    def _process_openai_image(self, file_path: str, prompt: str) -> str:
        """Process an image using the OpenAI Responses API.

        Args:
            file_path: Path to image file.
            prompt: Prompt text.

        Returns:
            Model response text.
        """
        with open(file_path, "rb") as f:
            result = self.client.files.create(
                file=f,
                purpose="vision",
            )
        image_part = {"type": "input_image", "file_id": result.id}

        response = self.client.responses.create(
            model=self.model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    image_part,
                ],
            }],
        )
        return (response.output_text or "").strip()

    def _process_openai_audio(self, file_path: str) -> str:
        """Process audio/video using OpenAI transcriptions.

        Args:
            file_path: Path to audio or video file.

        Returns:
            Transcription text or an error message.
        """
        ext = os.path.splitext(file_path)[1].lower()
        allowed_exts = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        if ext not in allowed_exts:
            return "Unsupported file type for OpenAI transcription."

        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:
            return "OpenAI transcription file size limit is 25 MB."

        with open(file_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model=self.transcription_model,
                file=f,
            )

        transcript_text = getattr(transcription, "text", None)
        if not transcript_text:
            transcript_text = str(transcription)

        return transcript_text.strip()

    def get_file_info(self, file_path: str) -> Dict[str, str]:
        """Return file metadata including MIME type and modality.

        Args:
            file_path: Path to media file.

        Returns:
            Metadata dictionary for the file.
        """
        mime_type, encoding = mimetypes.guess_type(file_path)
        modality = self._detect_modality(file_path)
        file_size = os.path.getsize(file_path)
        
        return {
            'file_path': file_path,
            'mime_type': mime_type or 'unknown',
            'encoding': encoding or 'none',
            'modality': modality,
            'file_size': f"{file_size:,} bytes",
            'processing_method': 'Files API' if file_size > 20 * 1024 * 1024 else 'Inline'
        }

def initialize_llm_model(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Any:
    """Initialize an LLM model for smolagents.

    Args:
        provider: "gemini" or "openai".
        model_name: Specific model name (required).
        temperature: Sampling temperature.
        **kwargs: Additional model-specific parameters.

    Returns:
        Initialized model instance compatible with smolagents.

    Raises:
        ValueError: If provider or model configuration is invalid.
    """
    # Provider configuration
    config = {
        "gemini": {
            "env_var": "GOOGLE_API_KEY",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/"
        },
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "api_base": None  # Uses OpenAI default
        }
    }
    
    if provider not in config:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {', '.join(config.keys())}")
    
    if not model_name:
        raise ValueError(f"model_name is required for provider '{provider}'")
    
    # Get API key
    api_key = os.environ.get(config[provider]["env_var"])
    if not api_key:
        raise ValueError(f"{config[provider]['env_var']} not found in environment")
    
    # Prepare kwargs
    model_kwargs = dict(kwargs)
    api_base = model_kwargs.pop("api_base", config[provider]["api_base"])
    top_p = model_kwargs.pop("top_p", 1.0)
    
    # Setup client kwargs
    client_kwargs = model_kwargs.pop("client_kwargs", {}) or {}
    client_kwargs.setdefault("timeout", 60)
    client_kwargs.setdefault("max_retries", 2)
    
    # Gemini: let the API default tool behavior unless explicitly set
    
    # Build arguments
    init_args = {
        "model_id": model_name,
        "api_key": api_key,
        "client_kwargs": client_kwargs,
        "temperature": temperature,
        **model_kwargs
    }
    
    # Add api_base if specified
    if api_base:
        init_args["api_base"] = api_base
        
    model = OpenAIServerModel(**init_args)
    global _FORMAT_PROVIDER, _FORMAT_MODEL_NAME
    _FORMAT_PROVIDER = provider
    _FORMAT_MODEL_NAME = model_name
    return model


class GAIAAgent:
    """GAIA agent using smolagents with Langfuse observability."""

    def __init__(
        self,
        user_id: str = None,
        session_id: str = None,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None
    ):
        """Initialize the agent with an LLM provider and MCP tools.

        Args:
            user_id: User identifier for tracking.
            session_id: Session identifier for tracking.
            provider: LLM provider ("gemini", "openai").
            model_name: Specific model name (uses defaults if None).
            mcp_servers: List of MCP server names to load.
        """

        # Initialize Langfuse observability
        self._setup_langfuse_observability()

        # Initialize LLM model
        self.model = initialize_llm_model(provider=provider, model_name=model_name, temperature=0.0)

        # Store user and session IDs for tracking
        self.user_id = user_id or "gaia-user"
        self.session_id = session_id or "gaia-session"

        # Initialize multimodal tool (optional)
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

        if api_key:
            self.multimodal_tool = UnifiedMultimodalTool(
                provider=provider,
                model_name=model_name or self.model.model_id,
                api_key=api_key
            )
        else:
            self.multimodal_tool = None
            logger.warning("Multimodal tool disabled; missing API key for provider %s.", provider)

        # GAIA system prompt
        self.system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts. 
                IMPORTANT:
                - In the last step of your reasoning, if you think your reasoning is not able to answer the question, answer the question directly with your internal reasoning, without using the visit_webpage tool.
                - Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
                """

        # Load MCP tools if specified
        self.mcp_tools = []
        if mcp_servers:
            from mcp_connectors import load_multiple_mcp_servers
            mcp_collections = load_multiple_mcp_servers(mcp_servers, trust_remote_code=True)
            for collection in mcp_collections:
                self.mcp_tools.extend(collection.tools)
            logger.info("Loaded %s MCP tools from %s servers", len(self.mcp_tools), len(mcp_collections))

        # Create the agent
        self.agent = None
        self._create_agent()

        # Initialize Langfuse client
        self.langfuse = None
        self.last_trace_id = None
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse()
        except Exception as e:
            logger.warning("Langfuse client unavailable: %s", e)

    def _setup_langfuse_observability(self):
        """Set up Langfuse observability with OpenTelemetry."""
        # Get Langfuse keys from environment variables
        langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        
        if not langfuse_public_key or not langfuse_secret_key:
            logger.warning("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not found. Observability will be limited.")
            return

        # Set up Langfuse environment variables
        os.environ["LANGFUSE_HOST"] = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        langfuse_auth = base64.b64encode(
            f"{langfuse_public_key}:{langfuse_secret_key}".encode()
        ).decode()
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

        # Create a TracerProvider for OpenTelemetry
        trace_provider = TracerProvider()
        
        # Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
        trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        
        # Set the global default tracer provider
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument smolagents with the configured provider
        SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    def _create_agent(self):
        """Create the CodeAgent with tools."""
        base_tools = [
            WebSearchTool(),
            visit_webpage,
            FinalAnswerTool()
        ]

        if self.multimodal_tool is not None:
            base_tools.append(self.multimodal_tool)

        all_tools = base_tools + self.mcp_tools

        self.agent = CodeAgent(
            tools=all_tools,
            model=self.model,
            additional_authorized_imports=[
                "math", "statistics", "itertools", "datetime",
                "random", "re", "json", "csv", "os", "sys",
                "collections", "functools", "pathlib",
                "numpy", "pandas", "matplotlib", "seaborn",
                "scipy", "sklearn", "requests", "bs4",
                "PIL", "yaml", "tqdm"
            ],
            max_steps=6
        )

    def _log_run_trace(self, prompt: str, response: str) -> Optional[str]:
        """Log input/output to Langfuse and return a trace id if available.

        Args:
            prompt: Input prompt text.
            response: Model response text.

        Returns:
            Trace id if recorded, otherwise None.
        """
        if not self.langfuse:
            return None

        metadata = {
            "provider": getattr(self.model, "provider", None),
            "model_id": getattr(self.model, "model_id", None),
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

        try:
            trace = self.langfuse.trace(
                name="chat",
                input=prompt,
                output=response,
                metadata=metadata,
            )
            self.langfuse.flush()
            return trace.id
        except Exception as e:
            logger.exception("Langfuse trace creation failed: %s", e)
            return None


    def _gaia_file_to_context(self, file_path: str) -> str:
        """Convert a GAIA task file into prompt context.

        Args:
            file_path: Path to the GAIA task file.

        Returns:
            Prompt-ready context string.
        """
        if not file_path:
            return ""

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or ""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if mime_type.startswith("image") or ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
            return (
                f"Attached media file: {file_path} (image).\n"
                "Use the multimodal_processor tool on this file_path before answering."
            )

        if mime_type.startswith("video") or ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return (
                f"Attached media file: {file_path} (video).\n"
                "Use the multimodal_processor tool on this file_path before answering."
            )

        if mime_type.startswith("audio") or ext in {".mp3", ".wav", ".m4a"}:
            return (
                f"Attached media file: {file_path} (audio).\n"
                "Use the multimodal_processor tool on this file_path before answering."
            )

        try:
            from llama_index.readers.docling import DoclingReader

            docling_reader = DoclingReader(
                export_type=DoclingReader.ExportType.MARKDOWN
            )
            documents = docling_reader.load_data(file_path)
            content = "\n\n".join([doc.text for doc in documents if doc.text])
            if content:
                return f"Document context from {filename}:\n{content}"
        except Exception as e:
            logger.exception("DoclingReader failed for %s: %s", filename, e)

        return ""


    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download a GAIA task file and return its local path.

        Args:
            task_id: GAIA task identifier.
            api_url: Base API URL.

        Returns:
            Local file path, or None on failure.
        """
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()

            # Try to get filename from headers
            logger.debug("Response headers: %s", response.headers)
            content_disp = response.headers.get("content-disposition", "")
            match = re.search(r'filename="(.+)"', content_disp)
            if match:
                filename = match.group(1)
            else:
                # Error
                raise ValueError("Filename not found in response headers")

            # Save the file
            with open(filename, 'wb') as f:
                f.write(response.content)

            logger.info("Downloaded file saved as %s", filename)
            return os.path.abspath(filename)  # Or just return `filename` for relative path

        except Exception as e:
            logger.exception("Failed to download file for task %s: %s", task_id, e)
            return None

    def run(self, query: str, max_steps: Optional[int] = None, reset_documents: bool = False) -> tuple[str, Optional[str]]:
        """Run the agent on a user query.

        Args:
            query: User question or instruction.
            max_steps: Maximum reasoning steps (overrides agent default).
            reset_documents: Reserved for compatibility (no-op).

        Returns:
            Tuple of (response_text, trace_id).
        """
        try:
            full_query = query

            if max_steps:
                original_max_steps = self.agent.max_steps
                self.agent.max_steps = max_steps

            response = self.agent.run(full_query)

            if max_steps:
                self.agent.max_steps = original_max_steps

            response_text = str(response)
            trace_id = self._log_run_trace(full_query, response_text)
            self.last_trace_id = trace_id
            return response_text, trace_id

        except Exception as e:
            error_msg = f"Error during agent execution: {e}"
            logger.exception("Error during agent execution: %s", e)
            import traceback
            traceback.print_exc()
            self.last_trace_id = None
            return error_msg, None

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from an agent response.

        Args:
            response: Full model response text.

        Returns:
            Extracted final answer string.
        """
        if "FINAL ANSWER:" in response:
            return response.split("FINAL ANSWER:")[-1].strip()

        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else response

    def solve_gaia_question(self, question_dict: Dict[str, Any]) -> str:
        """Solve a GAIA benchmark question (legacy method for evaluation).

        Args:
            question_dict: Dictionary with "Question" key and optional "task_id".

        Returns:
            Formatted answer for GAIA benchmark.
        """
        question_text = question_dict.get("Question", "")
        task_id = question_dict.get("task_id")

        if task_id:
            file_path = self.download_gaia_file(task_id)
            if file_path:
                context = self._gaia_file_to_context(file_path)
                if context:
                    question_text = f"{context}\n\nQuestion: {question_text}"

        response, _ = self.run(question_text)
        final_answer = self._extract_final_answer(response)

        return final_answer


    def add_user_feedback(self, trace_id: str, feedback_score: int, comment: str = None):
        """Add user feedback to a specific trace.

        Args:
            trace_id: Trace id to score.
            feedback_score: Score from 0-5 (0=very bad, 5=excellent).
            comment: Optional user comment.
        """
        try:
            self.langfuse.score(
                trace_id=trace_id,
                name="user-feedback",
                value=feedback_score,
                comment=comment
            )
            self.langfuse.flush()
            logger.info("User feedback added: %s/5", feedback_score)
        except Exception as e:
            logger.exception("Error adding user feedback: %s", e)
