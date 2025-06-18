"""
LLM-based reranking for GraphRAG relationship filtering.
"""
from __future__ import annotations

import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import AzureChatOpenAI


class RelationshipReranker:
    """LLM-based reranker for GraphRAG relationships."""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.0
    ):
        """
        Initialize relationship reranker with Azure OpenAI.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            deployment_name: Azure OpenAI deployment name
            api_version: API version
            temperature: Model temperature for generation
        """
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=deployment_name,
            api_version=api_version,
            temperature=temperature,
        )
        
        # One-shot learning examples
        self.one_shot_input = self._get_one_shot_input()
        self.one_shot_output = self._get_one_shot_output()
        
        # Query prompt template
        self.query_template = self._get_query_template()
    
    def _get_one_shot_input(self) -> str:
        """Get one-shot learning input example."""
        return """I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.

"""
    
    def _get_one_shot_output(self) -> str:
        """Get one-shot learning output example."""
        return """{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}"""
    
    def _get_query_template(self) -> str:
        """Get query prompt template."""
        return """Question:
{question}

Relationship descriptions:
{relation_des_str}

"""
    
    def rerank_relations(
        self,
        query: str,
        relation_candidate_texts: List[str],
        relation_candidate_ids: List[int],
        top_k: int = 3
    ) -> List[int]:
        """
        Rerank candidate relations using LLM Chain-of-Thought reasoning.
        
        Args:
            query: The input question
            relation_candidate_texts: List of candidate relationship descriptions
            relation_candidate_ids: List of corresponding relation IDs
            top_k: Number of top relationships to select
            
        Returns:
            List of reranked relation IDs in order of relevance
        """
        # Format relationship descriptions
        relation_des_str = "\n".join([
            f"[{rel_id}] {rel_text}"
            for rel_id, rel_text in zip(relation_candidate_ids, relation_candidate_texts)
        ]).strip()
        
        # Create prompt with one-shot learning
        rerank_prompts = ChatPromptTemplate.from_messages([
            HumanMessage(self.one_shot_input),
            AIMessage(self.one_shot_output),
            HumanMessagePromptTemplate.from_template(self.query_template),
        ])
        
        # Create chain with JSON output parser
        rerank_chain = (
            rerank_prompts
            | self.llm.bind(response_format={"type": "json_object"})
            | JsonOutputParser()
        )
        
        # Execute reranking
        try:
            rerank_res = rerank_chain.invoke({
                "question": query,
                "relation_des_str": relation_des_str
            })
            
            # Extract relation IDs from response
            rerank_relation_ids = []
            rerank_relation_lines = rerank_res.get("useful_relationships", [])
            
            for line in rerank_relation_lines:
                try:
                    # Extract ID from format "[ID] description"
                    start_idx = line.find("[") + 1
                    end_idx = line.find("]")
                    if start_idx > 0 and end_idx > start_idx:
                        rel_id = int(line[start_idx:end_idx])
                        rerank_relation_ids.append(rel_id)
                except (ValueError, IndexError):
                    continue
            
            return rerank_relation_ids[:top_k]
            
        except Exception as e:
            print(f"Error in LLM reranking: {e}")
            # Fallback: return first top_k candidates
            return relation_candidate_ids[:top_k]
    
    def generate_answer(
        self,
        query: str,
        context_passages: List[str]
    ) -> str:
        """
        Generate final answer using retrieved context.
        
        Args:
            query: The input question
            context_passages: List of relevant passages
            
        Returns:
            Generated answer
        """
        if not context_passages:
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join([
            f"Passage {i+1}: {passage}"
            for i, passage in enumerate(context_passages)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                """Use the following pieces of retrieved context to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.

Question: {question}

Context:
{context}

Answer:"""
            )
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({
                "question": query,
                "context": context_text
            })
            return answer.strip()
        except Exception as e:
            print(f"Error in answer generation: {e}")
            return "I encountered an error while generating the answer." 