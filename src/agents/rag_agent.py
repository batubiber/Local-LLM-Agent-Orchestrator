"""
RAGAgent implementation using Retrieval-Augmented Generation.
"""
from __future__ import annotations

import os
from typing import Any, List
from src.agents.base_agent import BaseAgent
from src.orchestrator.agent_factory import RAGStrategy
from src.data.vector_store import VectorStore
from src.data.document_processor import DocumentProcessor


class RAGAgent(BaseAgent):
    """Agent that retrieves relevant documents and generates answers."""

    def __init__(
        self,
        name: str,
        vector_store: VectorStore,
        llm: Any,
        context_dir: str = "context"
    ) -> None:
        """
        :param name: Unique name for the agent.
        :param vector_store: Vector store for document retrieval.
        :param llm: Language model for generation.
        :param context_dir: Directory containing context documents.
        """
        self._name = name
        self._vector_store = vector_store
        self._llm = llm
        self._strategy = RAGStrategy()
        self._context_dir = context_dir
        self._document_processor = DocumentProcessor()
        self._process_documents()

    def _process_documents(self) -> None:
        """Process all documents in the context directory."""
        if not os.path.exists(self._context_dir):
            return

        for filename in os.listdir(self._context_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self._context_dir, filename)
                try:
                    for chunk in self._document_processor.process_pdf(file_path):
                        # Compute embedding for the chunk
                        chunk = self._document_processor.compute_embedding(chunk)
                        # Add to vector store
                        self._vector_store.add_vectors(
                            vectors=[chunk.embedding],
                            metadata=[chunk.metadata]
                        )
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can handle the request."""
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in ['search', 'find', 'what does', 'summarize'])

    def process(self, request: str) -> dict:
        """Process the request via RAG strategy."""
        return self._strategy.execute(self, request)

    def retrieve_and_generate(self, request: str) -> dict:
        """Retrieve documents and generate a response using LLM."""
        try:
            # Retrieve top k contexts
            contexts = self._vector_store.query(request)
            
            if not contexts:
                return {
                    'answer': "I couldn't find any relevant information in the documents. Could you please provide more context or try a different query?",
                    'contexts': [],
                    'response': "I couldn't find any relevant information in the documents. Could you please provide more context or try a different query?"
                }

            # Prepare prompt
            context_text = "\n".join([f"Source: {ctx.get('source', 'Unknown')}, Page: {ctx.get('page', 'N/A')}\nContent: {ctx.get('text', '')}" for ctx in contexts])
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {request}

Answer:"""

            # Generate answer
            answer = self._llm.generate(prompt)
            
            return {
                'answer': answer,
                'contexts': contexts,
                'response': answer
            }
        except Exception as e:
            return {
                'answer': f"An error occurred while processing your request: {str(e)}",
                'contexts': [],
                'response': f"An error occurred while processing your request: {str(e)}"
            }

    def summarize(self, request: str) -> dict:
        """Generate a summary of the documents."""
        try:
            # Get all contexts
            contexts = self._vector_store.query(request, top_k=10)
            
            if not contexts:
                return {
                    'summary': "I couldn't find any documents to summarize. Please make sure there are documents in the context directory.",
                    'response': "I couldn't find any documents to summarize. Please make sure there are documents in the context directory."
                }

            # Prepare prompt
            context_text = "\n".join([f"Source: {ctx.get('source', 'Unknown')}, Page: {ctx.get('page', 'N/A')}\nContent: {ctx.get('text', '')}" for ctx in contexts])
            prompt = f"""Please provide a comprehensive summary of the following content:

{context_text}

Summary:"""

            # Generate summary
            summary = self._llm.generate(prompt)
            
            return {
                'summary': summary,
                'response': summary
            }
        except Exception as e:
            return {
                'summary': f"An error occurred while summarizing: {str(e)}",
                'response': f"An error occurred while summarizing: {str(e)}"
            }