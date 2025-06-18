"""
Triplet extraction for GraphRAG knowledge graph construction.
"""
from __future__ import annotations

import re
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI


@dataclass
class Triplet:
    """Represents a knowledge graph triplet (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    
    def to_sentence(self) -> str:
        """Convert triplet to natural language sentence."""
        return f"{self.subject} {self.predicate} {self.object}"
    
    def to_list(self) -> List[str]:
        """Convert triplet to list format."""
        return [self.subject, self.predicate, self.object]


class TripletExtractor:
    """Extract knowledge graph triplets from text using Azure OpenAI."""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.0
    ):
        """
        Initialize triplet extractor with Azure OpenAI.
        
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
            model_kwargs={
                "functions": [
                    {
                        "name": "extract_triplets",
                        "description": "Extract subject-predicate-object triplets from text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "triplets": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 3,
                                        "maxItems": 3
                                    },
                                    "description": "List of triplets, each containing [subject, predicate, object]"
                                }
                            },
                            "required": ["triplets"]
                        }
                    }
                ],
                "function_call": {"name": "extract_triplets"}
            }
        )
        
        self.extraction_prompt = self._create_extraction_prompt()
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for triplet extraction."""
        system_message = """You are a knowledge graph triplet extractor. Extract subject-predicate-object triplets from the given text.
Each triplet should be in the format [subject, predicate, object].

Example:
Input: "Albert Einstein was born in Germany in 1879. He developed the theory of relativity."
Expected triplets:
- ["Albert Einstein", "was born in", "Germany"]
- ["Albert Einstein", "was born in", "1879"]
- ["Albert Einstein", "developed", "theory of relativity"]"""

        human_message = """Text to extract triplets from:
{text}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def extract_triplets(self, text: str) -> List[Triplet]:
        """
        Extract triplets from text using LLM.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted Triplet objects
        """
        if not text or len(text.strip()) < 10:
            print("⚠️ Text is too short for triplet extraction")
            return []
            
        try:
            print("🔍 Extracting triplets using LLM...")
            
            # Format the messages
            messages = self.extraction_prompt.format_messages(text=text)
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Parse function call response
            try:
                if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                    function_call = response.additional_kwargs['function_call']
                    if function_call['name'] == 'extract_triplets':
                        data = json.loads(function_call['arguments'])
                        triplets_data = data.get('triplets', [])
                    else:
                        print("⚠️ Unexpected function call name")
                        return []
                else:
                    print("⚠️ No function call in response")
                    return []
                
            except Exception as e:
                print(f"⚠️ Error parsing LLM response: {e}")
                print(f"Raw response: {response}")
                return []
            
            if not triplets_data:
                print("⚠️ No triplets found in the text")
                return []
            
            # Convert to Triplet objects
            triplets = []
            for triplet_data in triplets_data:
                try:
                    if len(triplet_data) == 3:
                        triplet = Triplet(
                            subject=str(triplet_data[0]).strip(),
                            predicate=str(triplet_data[1]).strip(), 
                            object=str(triplet_data[2]).strip()
                        )
                        # Filter out empty or very short triplets
                        if all(len(part) > 1 for part in [triplet.subject, triplet.predicate, triplet.object]):
                            triplets.append(triplet)
                except Exception as e:
                    print(f"⚠️ Error processing triplet {triplet_data}: {e}")
                    continue
            
            print(f"✅ Extracted {len(triplets)} valid triplets")
            return triplets
            
        except Exception as e:
            print(f"❌ Error extracting triplets: {e}")
            return []
    
    def extract_entities_from_triplets(self, triplets: List[Triplet]) -> List[str]:
        """Extract unique entities from triplets."""
        entities = set()
        for triplet in triplets:
            entities.add(triplet.subject)
            entities.add(triplet.object)
        return list(entities)
    
    def build_adjacency_mappings(
        self, 
        triplets: List[Triplet],
        entities: List[str],
        relations: List[str]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Build adjacency mappings for graph traversal.
        
        Args:
            triplets: List of extracted triplets
            entities: List of unique entities
            relations: List of unique relations
            
        Returns:
            Tuple of (entityid_2_relationids, relationid_2_passageids)
        """
        entity_to_id = {entity: i for i, entity in enumerate(entities)}
        relation_to_id = {relation: i for i, relation in enumerate(relations)}
        
        entityid_2_relationids = {i: [] for i in range(len(entities))}
        
        for triplet in triplets:
            relation_sentence = triplet.to_sentence()
            if relation_sentence in relation_to_id:
                relation_id = relation_to_id[relation_sentence]
                
                # Map subject to relation
                if triplet.subject in entity_to_id:
                    subject_id = entity_to_id[triplet.subject]
                    if relation_id not in entityid_2_relationids[subject_id]:
                        entityid_2_relationids[subject_id].append(relation_id)
                
                # Map object to relation  
                if triplet.object in entity_to_id:
                    object_id = entity_to_id[triplet.object]
                    if relation_id not in entityid_2_relationids[object_id]:
                        entityid_2_relationids[object_id].append(relation_id)
        
        return entityid_2_relationids, {}


class SimpleRuleBasedExtractor:
    """Simple rule-based triplet extractor as fallback."""
    
    def extract_triplets(self, text: str) -> List[Triplet]:
        """
        Extract triplets using simple pattern matching.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted Triplet objects
        """
        triplets = []
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            # Simple patterns for common relationships
            patterns = [
                r'(.+?)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(.+)',
                r'(.+?)\s+(?:has|have|had)\s+(.+)',
                r'(.+?)\s+(?:born|died|lived)\s+(?:in|on|at)\s+(.+)',
                r'(.+?)\s+(?:worked|studied|taught)\s+(?:at|in|on)\s+(.+)',
                r'(.+?)\s+(?:wrote|created|developed|invented)\s+(.+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        subject = match[0].strip()
                        obj = match[1].strip()
                        
                        # Determine predicate based on pattern
                        if 'born' in sentence.lower():
                            predicate = 'was born in'
                        elif 'died' in sentence.lower():
                            predicate = 'died in'
                        elif 'worked' in sentence.lower():
                            predicate = 'worked at'
                        elif 'wrote' in sentence.lower():
                            predicate = 'wrote'
                        else:
                            predicate = 'is related to'
                        
                        if len(subject) > 1 and len(obj) > 1:
                            triplets.append(Triplet(subject, predicate, obj))
        
        return triplets
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10] 