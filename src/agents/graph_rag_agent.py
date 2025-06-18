"""
GraphRAG Agent implementation with Milvus Vector DB and Azure OpenAI.
"""
from __future__ import annotations

import os
import logging
from typing import Any, List, Dict, Optional
from collections import defaultdict

from src.agents.base_agent import BaseAgent
from src.orchestrator.agent_factory import RAGStrategy
from src.data.milvus_store import MilvusGraphStore
from src.data.graph_expansion import GraphExpander, NamedEntityRecognizer
from src.data.llm_reranker import RelationshipReranker
from src.data.triplet_extractor import TripletExtractor, SimpleRuleBasedExtractor
from src.data.document_processor import DocumentProcessor

# Set up logging
logger = logging.getLogger(__name__)

class GraphRAGAgent(BaseAgent):
    """GraphRAG agent with Milvus vector store and Azure OpenAI integration."""

    def __init__(
        self,
        name: str,
        milvus_uri: str,
        milvus_token: Optional[str],
        azure_endpoint: str,
        azure_api_key: str,
        azure_deployment: str,
        context_dir: str = "context",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.0,
        use_llm_extraction: bool = True
    ) -> None:
        """
        Initialize GraphRAG agent.

        Args:
            name: Agent name
            milvus_uri: Milvus connection URI
            milvus_token: Milvus authentication token
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            context_dir: Directory containing documents
            embedding_model: Sentence transformer model
            api_version: Azure OpenAI API version
            temperature: Model temperature
            use_llm_extraction: Whether to use LLM for triplet extraction
        """
        self._name = name
        self._context_dir = context_dir
        self._strategy = RAGStrategy()
        
        # Initialize components
        self.vector_store = MilvusGraphStore(
            uri=milvus_uri,
            token=milvus_token,
            embedding_model=embedding_model
        )
        
        self.reranker = RelationshipReranker(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            deployment_name=azure_deployment,
            api_version=api_version,
            temperature=temperature
        )
        
        if use_llm_extraction:
            self.triplet_extractor = TripletExtractor(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                deployment_name=azure_deployment,
                api_version=api_version,
                temperature=temperature
            )
        else:
            self.triplet_extractor = SimpleRuleBasedExtractor()
        
        self.document_processor = DocumentProcessor(embedding_model=embedding_model)
        
        # Data structures
        self.entities = []
        self.relations = []
        self.passages = []
        self.entityid_2_relationids = defaultdict(list)
        self.relationid_2_passageids = defaultdict(list)
        
        # Initialize graph components
        self.graph_expander = None
        self.ner = None
        
        # Process documents and build knowledge graph
        self._build_knowledge_graph()

    def _build_knowledge_graph(self) -> None:
        """Build knowledge graph from documents in context directory."""
        if not os.path.exists(self._context_dir):
            logger.error(f"Context directory {self._context_dir} does not exist")
            return

        logger.info("Building knowledge graph from documents...")
        
        # Process each document
        all_triplets = []
        processed_files = 0
        
        for filename in os.listdir(self._context_dir):
            if filename.endswith(('.pdf', '.txt')):
                file_path = os.path.join(self._context_dir, filename)
                passage_id = len(self.passages)
                
                try:
                    # Extract text from document
                    if filename.endswith('.pdf'):
                        logger.info(f"Processing PDF file: {filename}")
                        try:
                            chunks = list(self.document_processor.process_pdf(file_path))
                            if not chunks:
                                logger.warning(f"No text chunks extracted from {filename}")
                                continue
                            full_text = "\n\n".join([chunk.text for chunk in chunks])
                            logger.info(f"Successfully extracted {len(chunks)} chunks from {filename}")
                        except Exception as e:
                            logger.error(f"Failed to process PDF {filename}: {e}")
                            continue
                    else:  # .txt file
                        logger.info(f"Processing text file: {filename}")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                full_text = f.read()
                            logger.info(f"Successfully read text file {filename}")
                        except Exception as e:
                            logger.error(f"Failed to read text file {filename}: {e}")
                            continue
                    
                    if not full_text.strip():
                        logger.warning(f"No content extracted from {filename}")
                        continue
                    
                    self.passages.append(full_text)
                    processed_files += 1
                    
                    logger.info(f"Extracting triplets from {filename}...")
                    
                    # Extract triplets from text
                    triplets = self.triplet_extractor.extract_triplets(full_text)
                    
                    if triplets:
                        logger.info(f"Extracted {len(triplets)} triplets from {filename}")
                        
                        # Store passage-triplet mappings
                        for triplet in triplets:
                            relation_sentence = triplet.to_sentence()
                            if relation_sentence not in self.relations:
                                self.relations.append(relation_sentence)
                            
                            relation_id = self.relations.index(relation_sentence)
                            if passage_id not in self.relationid_2_passageids[relation_id]:
                                self.relationid_2_passageids[relation_id].append(passage_id)
                        
                        all_triplets.extend(triplets)
                    else:
                        logger.warning(f"No triplets extracted from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue

        if not all_triplets:
            logger.error("No triplets extracted from any documents")
            return

        logger.info(f"Successfully processed {processed_files} files")

        # Extract entities from triplets
        entity_set = set()
        for triplet in all_triplets:
            entity_set.add(triplet.subject)
            entity_set.add(triplet.object)
        
        self.entities = list(entity_set)
        
        # Build entity-relation mappings
        entity_to_id = {entity: i for i, entity in enumerate(self.entities)}
        relation_to_id = {relation: i for i, relation in enumerate(self.relations)}
        
        for triplet in all_triplets:
            relation_sentence = triplet.to_sentence()
            if relation_sentence in relation_to_id:
                relation_id = relation_to_id[relation_sentence]
                
                # Map subject to relation
                if triplet.subject in entity_to_id:
                    subject_id = entity_to_id[triplet.subject]
                    if relation_id not in self.entityid_2_relationids[subject_id]:
                        self.entityid_2_relationids[subject_id].append(relation_id)
                
                # Map object to relation
                if triplet.object in entity_to_id:
                    object_id = entity_to_id[triplet.object]
                    if relation_id not in self.entityid_2_relationids[object_id]:
                        self.entityid_2_relationids[object_id].append(relation_id)

        logger.info(f"Extracted {len(self.entities)} entities, {len(self.relations)} relations from {len(self.passages)} passages")

        # Insert data into Milvus
        logger.info("Inserting data into Milvus...")
        print("Inserting data into Milvus...")
        
        print(f"Inserting {len(self.entities)} entities...")
        self.vector_store.insert_data(
            entities=self.entities,
            relations=self.relations,
            passages=self.passages,
            entityid_2_relationids=self.entityid_2_relationids,
            relationid_2_passageids=self.relationid_2_passageids
        )

        # Initialize graph components
        self.graph_expander = GraphExpander(
            entities=self.entities,
            relations=self.relations,
            entityid_2_relationids=self.entityid_2_relationids
        )
        
        self.ner = NamedEntityRecognizer(entities=self.entities)
        
        logger.info("Knowledge graph built successfully!")
        print("Knowledge graph built successfully!")

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can handle the request."""
        # GraphRAG can handle any query that requires reasoning
        return True

    def process(self, request: str) -> dict:
        """Process the request via GraphRAG strategy."""
        return self._strategy.execute(self, request)

    def retrieve_and_generate(self, request: str) -> dict:
        """
        Perform GraphRAG retrieval and generation.
        
        Args:
            request: User query
            
        Returns:
            Response dictionary with answer and context
        """
        try:
            if not self.graph_expander or not self.ner:
                return {
                    'answer': "Knowledge graph not initialized. Please ensure documents are processed.",
                    'contexts': [],
                    'response': "Knowledge graph not initialized. Please ensure documents are processed."
                }

            print(f"Processing query: {request}")

            # Step 1: Entity extraction and similarity search
            query_entities = self.ner.extract_entities(request)
            print(f"Extracted entities: {query_entities}")
            
            top_k = 3
            
            # Search for similar entities
            entity_search_results = []
            if query_entities:
                entity_search_results = self.vector_store.search_entities(
                    query_texts=query_entities, 
                    top_k=top_k
                )

            # Search for similar relations
            relation_search_results = self.vector_store.search_relations(
                query_text=request, 
                top_k=top_k
            )

            print(f"Found {len(entity_search_results)} entity results, {len(relation_search_results)} relation results")

            # Step 2: Graph expansion
            candidate_relation_ids = self.graph_expander.expand_subgraph(
                entity_search_results=entity_search_results,
                relation_search_results=relation_search_results,
                target_degree=1
            )

            if not candidate_relation_ids:
                return {
                    'answer': "No relevant relationships found in the knowledge graph.",
                    'contexts': [],
                    'response': "No relevant relationships found in the knowledge graph."
                }

            print(f"Expanded to {len(candidate_relation_ids)} candidate relations")

            # Get candidate relation texts
            candidate_relation_texts = self.vector_store.get_relations_by_ids(candidate_relation_ids)

            # Step 3: LLM reranking
            reranked_relation_ids = self.reranker.rerank_relations(
                query=request,
                relation_candidate_texts=candidate_relation_texts,
                relation_candidate_ids=candidate_relation_ids,
                top_k=3
            )

            print(f"Reranked to {len(reranked_relation_ids)} top relations")

            # Step 4: Retrieve final passages
            final_passage_ids = []
            for relation_id in reranked_relation_ids:
                passage_ids = self.relationid_2_passageids.get(relation_id, [])
                for passage_id in passage_ids:
                    if passage_id not in final_passage_ids:
                        final_passage_ids.append(passage_id)

            final_passages = [
                self.passages[pid] for pid in final_passage_ids 
                if pid < len(self.passages)
            ][:2]  # Limit to top 2 passages

            print(f"Retrieved {len(final_passages)} final passages")

            # Step 5: Generate answer
            if final_passages:
                answer = self.reranker.generate_answer(
                    query=request,
                    context_passages=final_passages
                )
            else:
                answer = "I couldn't find relevant information to answer your question."

            # Prepare contexts for display
            contexts = [
                {
                    'text': passage,
                    'source': f'Document {i+1}',
                    'score': 1.0 - (i * 0.1)  # Mock scoring
                }
                for i, passage in enumerate(final_passages)
            ]

            return {
                'answer': answer,
                'contexts': contexts,
                'response': answer
            }

        except Exception as e:
            print(f"Error in GraphRAG processing: {e}")
            return {
                'answer': f"An error occurred while processing your request: {str(e)}",
                'contexts': [],
                'response': f"An error occurred while processing your request: {str(e)}"
            }

    def summarize(self, request: str) -> dict:
        """Generate a summary using GraphRAG approach."""
        try:
            # Use all passages for summarization
            if not self.passages:
                return {
                    'summary': "No documents available for summarization.",
                    'response': "No documents available for summarization."
                }

            # Combine all passages
            all_text = "\n\n".join(self.passages)
            
            summary = self.reranker.generate_answer(
                query="Provide a comprehensive summary of the following content:",
                context_passages=[all_text]
            )

            return {
                'summary': summary,
                'response': summary
            }

        except Exception as e:
            return {
                'summary': f"An error occurred while summarizing: {str(e)}",
                'response': f"An error occurred while summarizing: {str(e)}"
            }

    def compare_with_naive_rag(self, request: str) -> Dict[str, Any]:
        """Compare GraphRAG results with naive RAG baseline."""
        try:
            # GraphRAG result
            graph_result = self.retrieve_and_generate(request)
            
            # Naive RAG result
            naive_results = self.vector_store.search_passages(request, top_k=2)
            naive_passages = [res["entity"]["text"] for res in naive_results]
            
            naive_answer = self.reranker.generate_answer(
                query=request,
                context_passages=naive_passages
            )
            
            return {
                'graph_rag': {
                    'answer': graph_result['answer'],
                    'passages': [ctx['text'] for ctx in graph_result['contexts']]
                },
                'naive_rag': {
                    'answer': naive_answer,
                    'passages': naive_passages
                }
            }
            
        except Exception as e:
            return {
                'error': f"Comparison failed: {str(e)}"
            } 