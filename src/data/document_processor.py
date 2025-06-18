"""
Document processing utilities for RAG system.
"""
from __future__ import annotations
import pypdf
import os
import logging
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    metadata: Dict[str, str]
    embedding: Optional[List[float]] = None


class DocumentProcessor:
    """Handles document processing and chunking for RAG."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,  # Increased chunk size
        chunk_overlap: int = 100  # Increased overlap
    ):
        """
        Initialize document processor.

        Args:
            embedding_model: Name/path of the sentence transformer model
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer(embedding_model)

    def process_pdf(self, file_path: str) -> Iterator[TextChunk]:
        """
        Process a PDF file and yield text chunks with metadata.

        Args:
            file_path: Path to PDF file

        Yields:
            TextChunk objects
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            with open(file_path, "rb") as file:
                try:
                    pdf = pypdf.PdfReader(file)
                    total_pages = len(pdf.pages)
                    logger.info(f"PDF has {total_pages} pages")
                    
                    extracted_text = []
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            text = page.extract_text()
                            if text.strip():
                                extracted_text.append(text)
                                logger.info(f"Successfully extracted text from page {page_num}")
                            else:
                                logger.warning(f"No text content in page {page_num}")
                        except Exception as e:
                            logger.error(f"Error extracting text from page {page_num}: {e}")
                            continue
                    
                    if not extracted_text:
                        logger.error("No text could be extracted from any page")
                        return
                    
                    # Combine all text with proper spacing
                    full_text = "\n\n".join(extracted_text)
                    
                    metadata = {
                        "source": os.path.basename(file_path),
                        "pages": f"1-{total_pages}",
                        "type": "pdf"
                    }
                    
                    # Process the combined text
                    chunk_count = 0
                    for chunk in self._chunk_text(full_text, metadata):
                        chunk_count += 1
                        yield chunk
                    
                    logger.info(f"Generated {chunk_count} chunks from PDF")
                    
                except pypdf.errors.PdfReadError as e:
                    logger.error(f"PDF parsing error: {e}")
                    raise RuntimeError(f"Failed to parse PDF {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            raise RuntimeError(f"Failed to process PDF {file_path}: {e}")

    def process_text(self, text: str, metadata: Dict[str, str]) -> Iterator[TextChunk]:
        """
        Process raw text and yield chunks with metadata.

        Args:
            text: Input text
            metadata: Metadata dictionary

        Yields:
            TextChunk objects
        """
        yield from self._chunk_text(text, metadata)

    def _chunk_text(self, text: str, metadata: Dict[str, str]) -> Iterator[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Yields:
            TextChunk objects
        """
        text = text.strip()
        if not text:
            return

        # First split by paragraphs
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If this paragraph alone exceeds chunk size, split it by sentences
            if para_length > self.chunk_size:
                # First yield the current chunk if it exists
                if current_chunk:
                    yield TextChunk(
                        text="\n\n".join(current_chunk),
                        metadata=metadata.copy(),
                        embedding=None
                    )
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = para.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|")
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    if len(sentence) > self.chunk_size:
                        # If a single sentence is too long, split by space
                        words = sentence.split()
                        word_chunk = []
                        word_length = 0
                        
                        for word in words:
                            if word_length + len(word) + 1 <= self.chunk_size:
                                word_chunk.append(word)
                                word_length += len(word) + 1
                            else:
                                if word_chunk:
                                    yield TextChunk(
                                        text=" ".join(word_chunk),
                                        metadata=metadata.copy(),
                                        embedding=None
                                    )
                                word_chunk = [word]
                                word_length = len(word) + 1
                        
                        if word_chunk:
                            yield TextChunk(
                                text=" ".join(word_chunk),
                                metadata=metadata.copy(),
                                embedding=None
                            )
                    else:
                        if sentence_length + len(sentence) + 2 <= self.chunk_size:
                            sentence_chunk.append(sentence)
                            sentence_length += len(sentence) + 2
                        else:
                            if sentence_chunk:
                                yield TextChunk(
                                    text=". ".join(sentence_chunk) + ".",
                                    metadata=metadata.copy(),
                                    embedding=None
                                )
                            sentence_chunk = [sentence]
                            sentence_length = len(sentence) + 2
                
                if sentence_chunk:
                    yield TextChunk(
                        text=". ".join(sentence_chunk) + ".",
                        metadata=metadata.copy(),
                        embedding=None
                    )
            
            # Normal case: add paragraph to current chunk
            elif current_length + para_length + 2 <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_length + 2
            else:
                # Yield current chunk and start new one
                if current_chunk:
                    yield TextChunk(
                        text="\n\n".join(current_chunk),
                        metadata=metadata.copy(),
                        embedding=None
                    )
                current_chunk = [para]
                current_length = para_length
        
        # Yield any remaining chunk
        if current_chunk:
            yield TextChunk(
                text="\n\n".join(current_chunk),
                metadata=metadata.copy(),
                embedding=None
            )

    def compute_embedding(self, chunk: TextChunk) -> TextChunk:
        """
        Compute embedding for a text chunk.

        Args:
            chunk: TextChunk object

        Returns:
            TextChunk with computed embedding
        """
        if not chunk.embedding:
            chunk.embedding = self.embedding_model.encode(
                chunk.text,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
        return chunk 