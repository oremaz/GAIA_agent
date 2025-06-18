import os
import requests
import base64
from typing import Dict, Any, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, OpenAIServerModel, Tool
from smolagents import PythonInterpreterTool, SpeechToTextTool

# Langfuse observability imports
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
from langfuse import Langfuse
from smolagents import PythonInterpreterTool, FinalAnswerTool, SpeechToTextTool


import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
import re
import mimetypes


from concurrent.futures import ThreadPoolExecutor, TimeoutError


class WebSearchTool(Tool):
    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def _perform_search(self, query: str):
        """Internal method to perform the actual search."""
        return self.ddgs.text(query, max_results=self.max_results)

    def forward(self, query: str) -> str:
        results = []
        
        # First attempt with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self._perform_search, query)
                results = future.result(timeout=30)  # 30 second timeout
            except TimeoutError:
                print("First search attempt timed out after 30 seconds, retrying...")
                results = []
        
        # Retry if no results or timeout occurred
        if len(results) == 0:
            print("Retrying search...")
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
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
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
    

class BM25RetrieverTool(Tool):
    """
    BM25 retriever tool for document search when text documents are available
    """
    name = "bm25_retriever"
    description = "Uses BM25 search to retrieve relevant parts of uploaded documents. Use this when the question references an attached file or document (except for images)."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant document sections.",
        }
    }
    output_type = "string"

    def __init__(self, docs=None, **kwargs):
        super().__init__(**kwargs)
        self.docs = docs or []
        self.retriever = None
        if self.docs:
            self.retriever = BM25Retriever.from_documents(self.docs, k=5)

    def forward(self, query: str) -> str:
        if not self.retriever:
            return "No documents loaded for retrieval."

        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join([
            f"\n\n===== Document {str(i)} =====\n" + doc.page_content
            for i, doc in enumerate(docs)
        ])

class GAIAAgent:
    """
    GAIA agent using smolagents with Gemini 2.0 Flash and Langfuse observability
    """

    def __init__(self, user_id: str = None, session_id: str = None):
        """Initialize the agent with Gemini 2.0 Flash, tools, and Langfuse observability"""

        # Get API keys
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")

        # Initialize Langfuse observability
        self._setup_langfuse_observability()

        # Initialize Gemini 2.0 Flash model
        self.model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key,
            temperature=0.0, 
            top_p=1.0,
        )

        # Store user and session IDs for tracking
        self.user_id = user_id or "gaia-user"
        self.session_id = session_id or "gaia-session"

        # GAIA system prompt from the leaderboard
        self.system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
                IMPORTANT:
                - In the last step of your reasoning, if you think your reasoning is not able to answer the question, answer the question directy with your internal reasoning, without using the BM25 retriever tool or the visit_webpage tool.
                - Always use the final_answer tool to return your final answer, even if you think you can answer the question without using the tools.
                """

        # Initialize retriever tool (will be updated when documents are loaded)
        self.retriever_tool = BM25RetrieverTool()

        # Create the agent
        self.agent = None
        self._create_agent()

        # Initialize Langfuse client
        self.langfuse = Langfuse()

        from langfuse import get_client
        self.langfuse = get_client()  # ✅ Use get_client() for v3
        
        # Store user and session IDs for tracking
        self.user_id = user_id or "gaia-user"
        self.session_id = session_id or "gaia-session"

    def _setup_langfuse_observability(self):
        """Set up Langfuse observability with OpenTelemetry"""
        # Get Langfuse keys from environment variables
        langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        
        if not langfuse_public_key or not langfuse_secret_key:
            print("Warning: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not found. Observability will be limited.")
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
        """Create the CodeAgent with tools"""
        base_tools = [
            self.retriever_tool, 
            visit_webpage,
        ]
        self.agent = CodeAgent(
            tools=base_tools + [
                WebSearchTool(),
                PythonInterpreterTool(),
                FinalAnswerTool(), 
                SpeechToTextTool()],
            model=self.model,
            description=self.system_prompt, 
            max_steps=5, 
            additional_authorized_imports = [
    "math",             # basic calculations
    "statistics",       # common numeric helpers
    "itertools",        # safe functional helpers
    "datetime",         # date handling
    "random",           # simple randomness (no os access)
    "re",               # regular expressions
    "json",             # serialisation / parsing
]       )



    def load_documents_from_file(self, file_path: str):
        """Load and process text documents for BM25, or return raw content for media files."""
        try:
            # Devine le type MIME du fichier
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = ""
            print(f"Detected MIME type: {mime_type} for file {file_path}")
            # Traitement selon le type de fichier
            if mime_type.startswith("image") or mime_type.startswith("video") or mime_type.startswith("audio"):
                with open(file_path, "rb") as f:
                    binary_content = f.read()
                print(f"Loaded {mime_type} file: {file_path}")
                return binary_content
            else : 
            # Si fichier texte → traitement BM25
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            # Split en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk, metadata={"source": file_path})
                    for chunk in chunks]

            # BM25 + agent
            self.retriever_tool = BM25RetrieverTool(docs)
            self._create_agent()

            print(f"Loaded {len(docs)} document chunks from {file_path}")
            return True

        except Exception as e:
            print(f"Error loading documents from {file_path}: {e}")
            return False


    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with GAIA task_id and return its path"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()

            # Try to get filename from headers
            print(f"Response headers: {response.headers}")
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

            print(f"Downloaded file saved as {filename}")
            return os.path.abspath(filename)  # Or just return `filename` for relative path

        except Exception as e:
            print(f"Failed to download file for task {task_id}: {e}")
            return None

    def solve_gaia_question(self, question_data: Dict[str, Any], tags: List[str] = None) -> str:
        """
        Solve a GAIA question with full Langfuse observability
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        
        # Prepare tags for observability
        trace_tags = ["gaia-agent", "question-solving"]
        if tags:
            trace_tags.extend(tags)
        if task_id:
            trace_tags.append(f"task-{task_id}")

        # Use SDK v3 context manager approach
        with self.langfuse.start_as_current_span(
            name="GAIA-Question-Solving",
            input={"question": question, "task_id": task_id},
            metadata={
                "model": self.model.model_id,
                "question_length": len(question),
                "has_file": bool(task_id)
            }
        ) as span:
            try:
                # Set trace attributes using v3 syntax
                span.update_trace(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    tags=trace_tags
                )

                # Download and load file if task_id provided
                file_loaded = False
                if task_id:
                    file_path = self.download_gaia_file(task_id)
                    if file_path:
                        file_loaded = self.load_documents_from_file(file_path)
                        print(f"Loaded file for task {task_id}")

                # Prepare the prompt
                prompt = f"""
    Question: {question}
    {f'Task ID: {task_id}' if task_id else ''}
    {f'File loaded: Yes as {file_path}' if file_loaded else 'File loaded: No'}

                """

                print("=== AGENT REASONING ===")
                result = self.agent.run(prompt)
                print("=== END REASONING ===")

                # Update span with result using v3 syntax
                span.update(output={"answer": str(result)})

                return str(result)

            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                print(error_msg)
                
                # Log error using v3 syntax
                span.update(
                    output={"error": error_msg},
                    level="ERROR"
                )
                
                return error_msg


    def evaluate_answer(self, question: str, answer: str, expected_answer: str = None) -> Dict[str, Any]:
        """
        Evaluate the agent's answer using LLM-as-a-Judge and optionally compare with expected answer
        """
        evaluation_prompt = f"""
Please evaluate the following answer to a question on a scale of 1-5:

Question: {question}
Answer: {answer}
{f'Expected Answer: {expected_answer}' if expected_answer else ''}

Rate the answer on:
1. Accuracy (1-5)
2. Completeness (1-5) 
3. Clarity (1-5)

Provide your rating as JSON: {{"accuracy": X, "completeness": Y, "clarity": Z, "overall": W, "reasoning": "explanation"}}
        """

        try:
            # Use the same model to evaluate
            evaluation_result = self.agent.run(evaluation_prompt)
            
            # Try to parse JSON response
            import json
            scores = json.loads(evaluation_result)
            return scores
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default structure
            print("Failed to parse evaluation result as JSON. Returning default scores.")
            return {
                "accuracy": 0,
                "completeness": 0,
                "clarity": 0,
                "overall": 0,
                "reasoning": "Could not parse evaluation result"
            }


    def add_user_feedback(self, trace_id: str, feedback_score: int, comment: str = None):
        """
        Add user feedback to a specific trace
        
        Args:
            trace_id: The trace ID to add feedback to
            feedback_score: Score from 0-5 (0=very bad, 5=excellent)
            comment: Optional comment from user
        """
        try:
            self.langfuse.score(
                trace_id=trace_id,
                name="user-feedback",
                value=feedback_score,
                comment=comment
            )
            self.langfuse.flush()
            print(f"User feedback added: {feedback_score}/5")
        except Exception as e:
            print(f"Error adding user feedback: {e}")


# Example usage with observability
if __name__ == "__main__":
    # Set up environment variables (you need to set these)
    # os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
    # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
    
    # Test the agent with observability
    agent = GAIAAgent(
        user_id="test-user-123",
        session_id="test-session-456"
    )

    # Example question
    question_data = {
        "Question": "Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3.  In your response, please only list the ingredients, not any measurements. So if the recipe calls for ""a pinch of salt"" or ""two cups of ripe strawberries"" the ingredients on the list would be ""salt"" and ""ripe strawberries"". Please format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.",
        "task_id": "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3"
    }

    # Solve with full observability
    answer = agent.solve_gaia_question(
        question_data, 
    )
    print(f"Answer: {answer}")