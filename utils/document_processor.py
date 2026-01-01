"""Document processing helpers for uploads, URLs, and Docling parsing."""

import os
import json
import logging
import tempfile
import uuid
from typing import List, Tuple, Optional, Dict, Callable
from pathlib import Path

from llama_index.core import Document
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import CSVReader, PandasExcelReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)


class UserAgentWebPageReader(SimpleWebPageReader):
    """SimpleWebPageReader with a configurable User-Agent header."""

    def __init__(
        self,
        html_to_text: bool = False,
        metadata_fn: Optional[Callable[[str], Dict]] = None,
        timeout: Optional[int] = 60,
        fail_on_error: bool = False,
        user_agent: Optional[str] = None,
    ) -> None:
        super().__init__(
            html_to_text=html_to_text,
            metadata_fn=metadata_fn,
            timeout=timeout,
            fail_on_error=fail_on_error,
        )
        self._user_agent = user_agent or DEFAULT_USER_AGENT

    def load_data(self, urls: List[str]) -> List[Document]:
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            try:
                import requests
                response = requests.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout,
                )
            except Exception:
                if self._fail_on_error:
                    raise
                continue

            response_text = response.text

            if response.status_code != 200 and self._fail_on_error:
                raise ValueError(
                    f"Error fetching page from {url}. server returned status:"
                    f" {response.status_code} and response {response_text}"
                )

            if self.html_to_text:
                import html2text

                response_text = html2text.html2text(response_text)

            metadata: Dict = {"url": url}
            if self._metadata_fn is not None:
                metadata = self._metadata_fn(url)
                if "url" not in metadata:
                    metadata["url"] = url

            documents.append(
                Document(text=response_text, id_=str(uuid.uuid4()), metadata=metadata)
            )

        return documents


class DocumentProcessor:
    """Process documents into LlamaIndex Document objects."""

    def __init__(self):
        """Initialize document readers and supported extensions."""

        # Initialize readers
        self.docling_reader = DoclingReader(
            export_type=DoclingReader.ExportType.MARKDOWN
        )
        self.csv_reader = CSVReader()
        self.excel_reader = PandasExcelReader()
        self.web_reader = UserAgentWebPageReader(
            html_to_text=True,
            fail_on_error=True,
        )
        self.youtube_reader = YoutubeTranscriptReader()

        # Supported extensions by reader
        self.docling_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.html', '.htm'}
        self.csv_extensions = {'.csv'}
        self.excel_extensions = {'.xlsx', '.xls'}
        self.json_extensions = {'.json'}
        self.text_extensions = {'.txt', '.md', '.markdown', '.rst'}

    def process_file(self, file_path: str) -> Tuple[List[Document], str]:
        """Process a file into Documents.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (documents, file_type).
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return [], "unknown"

        file_extension = Path(file_path).suffix.lower()
        file_type = self._get_file_type(file_extension)

        logger.info("Processing file: %s (type: %s)", file_path, file_type)

        try:
            if file_extension in self.docling_extensions:
                documents = self._process_with_docling(file_path)

            elif file_extension in self.csv_extensions:
                documents = self.csv_reader.load_data(file_path)

            elif file_extension in self.excel_extensions:
                documents = self.excel_reader.load_data(file_path)

            elif file_extension in self.json_extensions:
                documents = self._process_json(file_path)

            elif file_extension in self.text_extensions:
                documents = self._process_text(file_path)

            else:
                # Try as plain text fallback
                logger.warning("Unknown file type, trying plain text: %s", file_extension)
                documents = self._process_text(file_path)

            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = file_path
                doc.metadata["file_type"] = file_type
                doc.metadata["file_name"] = Path(file_path).name

            logger.info("Processed %d documents from %s", len(documents), file_path)
            return documents, file_type

        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, e)
            return [], file_type

    def process_uploaded_file(self, uploaded_file) -> Tuple[List[Document], str]:
        """Process a Streamlit uploaded file into Documents.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Tuple of (documents, file_type).
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            documents, file_type = self.process_file(tmp_path)

            # Update metadata with original filename
            for doc in documents:
                doc.metadata["file_name"] = uploaded_file.name

            return documents, file_type

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def process_url(self, url: str) -> Tuple[List[Document], str]:
        """Process a URL into Documents.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (documents, url_type).
        """
        logger.info("Processing URL: %s", url)

        try:
            from urllib.parse import urlparse
            netloc = urlparse(url).netloc.lower()

            # YouTube handling
            if "youtube" in netloc or "youtu.be" in netloc:
                logger.info("Processing YouTube video")
                documents = self.youtube_reader.load_data(youtubelinks=[url])
                url_type = "youtube"

            # Regular web page
            else:
                logger.info("Processing web page")
                documents = self.web_reader.load_data(urls=[url])
                if documents:
                    logger.info("Web page text length: %d", len(documents[0].text or ""))
                url_type = "webpage"

            # Add metadata
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["url_type"] = url_type

            logger.info("Processed %d documents from URL", len(documents))
            return documents, url_type

        except Exception as e:
            logger.error("Error processing URL %s: %s", url, e)
            return [], "unknown"


    def _process_with_docling(self, file_path: str) -> List[Document]:
        """Process a file using DoclingReader.

        Args:
            file_path: Path to the file.

        Returns:
            List of Documents parsed by Docling.
        """
        try:
            documents = self.docling_reader.load_data(file_path)
            return documents
        except Exception as e:
            logger.error("DoclingReader failed: %s", e)
            return []

    def _process_text(self, file_path: str) -> List[Document]:
        """Process a plain text file.

        Args:
            file_path: Path to the file.

        Returns:
            List of Documents parsed from the text file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc = Document(
                text=content,
                metadata={"source": file_path}
            )

            return [doc]

        except UnicodeDecodeError:
            # Try with latin-1 encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

                doc = Document(
                    text=content,
                    metadata={"source": file_path}
                )

                return [doc]

            except Exception as e:
                logger.error("Error reading text file: %s", e)
                return []

    def _process_json(self, file_path: str) -> List[Document]:
        """Process a JSON file into a single Document.

        Args:
            file_path: Path to the file.

        Returns:
            List containing a single Document on success, otherwise empty.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            content = json.dumps(
                data,
                ensure_ascii=True,
                indent=2,
                sort_keys=True
            )
            return [Document(text=content, metadata={"source": file_path})]
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Failed to read JSON %s: %s", file_path, e)
            return []

    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension.

        Args:
            extension: File extension including leading dot.

        Returns:
            File type label.
        """
        if extension in self.docling_extensions:
            return "document"
        if extension in self.csv_extensions:
            return "csv"
        if extension in self.excel_extensions:
            return "spreadsheet"
        if extension in self.json_extensions:
            return "json"
        if extension in self.text_extensions:
            return "text"
        return "unknown"
