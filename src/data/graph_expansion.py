"""
Graph expansion utilities for GraphRAG multi-hop reasoning.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Set
from collections import defaultdict


class GraphExpander:
    """Handles graph expansion operations using adjacency matrices."""
    
    def __init__(
        self,
        entities: List[str],
        relations: List[str],
        entityid_2_relationids: Dict[int, List[int]]
    ):
        """
        Initialize graph expander with entity-relation mappings.
        
        Args:
            entities: List of entity strings
            relations: List of relation strings
            entityid_2_relationids: Mapping from entity IDs to relation IDs
        """
        self.entities = entities
        self.relations = relations
        self.entityid_2_relationids = entityid_2_relationids
        
        # Build adjacency matrices
        self.entity_relation_adj = self._build_entity_relation_adjacency()
        self.entity_adj_1_degree = self._build_entity_adjacency()
        self.relation_adj_1_degree = self._build_relation_adjacency()
    
    def _build_entity_relation_adjacency(self) -> csr_matrix:
        """Build entity-relation adjacency matrix."""
        entity_relation_adj = np.zeros((len(self.entities), len(self.relations)))
        
        for entity_id, relation_ids in self.entityid_2_relationids.items():
            if entity_id < len(self.entities):
                for relation_id in relation_ids:
                    if relation_id < len(self.relations):
                        entity_relation_adj[entity_id, relation_id] = 1
        
        return csr_matrix(entity_relation_adj)
    
    def _build_entity_adjacency(self) -> csr_matrix:
        """Build entity-entity adjacency matrix (1-degree)."""
        return self.entity_relation_adj @ self.entity_relation_adj.T
    
    def _build_relation_adjacency(self) -> csr_matrix:
        """Build relation-relation adjacency matrix (1-degree)."""
        return self.entity_relation_adj.T @ self.entity_relation_adj
    
    def expand_subgraph(
        self,
        entity_search_results: List[List[Dict]],
        relation_search_results: List[Dict],
        target_degree: int = 1
    ) -> List[int]:
        """
        Expand subgraph from retrieved entities and relations.
        
        Args:
            entity_search_results: Results from entity similarity search
            relation_search_results: Results from relation similarity search
            target_degree: Degree of expansion (1 or 2 hop)
            
        Returns:
            List of candidate relation IDs
        """
        # Compute target degree adjacency matrices
        entity_adj_target_degree = self.entity_adj_1_degree
        for _ in range(target_degree - 1):
            entity_adj_target_degree = entity_adj_target_degree * self.entity_adj_1_degree
            
        relation_adj_target_degree = self.relation_adj_1_degree
        for _ in range(target_degree - 1):
            relation_adj_target_degree = relation_adj_target_degree * self.relation_adj_1_degree
        
        entity_relation_adj_target_degree = entity_adj_target_degree @ self.entity_relation_adj
        
        # Expand from relations
        expanded_relations_from_relation = set()
        filtered_hit_relation_ids = [
            relation_res["entity"]["id"] for relation_res in relation_search_results
        ]
        
        for hit_relation_id in filtered_hit_relation_ids:
            if hit_relation_id < relation_adj_target_degree.shape[0]:
                expanded_relations_from_relation.update(
                    relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
                )
        
        # Expand from entities
        expanded_relations_from_entity = set()
        filtered_hit_entity_ids = [
            one_entity_res["entity"]["id"]
            for one_entity_search_res in entity_search_results
            for one_entity_res in one_entity_search_res
        ]
        
        for hit_entity_id in filtered_hit_entity_ids:
            if hit_entity_id < entity_relation_adj_target_degree.shape[0]:
                expanded_relations_from_entity.update(
                    entity_relation_adj_target_degree[hit_entity_id].nonzero()[1].tolist()
                )
        
        # Merge expanded relations
        relation_candidate_ids = list(
            expanded_relations_from_relation | expanded_relations_from_entity
        )
        
        return relation_candidate_ids


class NamedEntityRecognizer:
    """Simple NER for extracting entities from queries."""
    
    def __init__(self, entities: List[str]):
        """
        Initialize with known entities.
        
        Args:
            entities: List of known entity strings
        """
        self.entities = set(entities)
        self.entity_lower = {entity.lower(): entity for entity in entities}
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from query text.
        
        Args:
            query: Input query string
            
        Returns:
            List of recognized entities
        """
        found_entities = []
        query_lower = query.lower()
        
        # Simple substring matching
        for entity_lower, entity_original in self.entity_lower.items():
            if entity_lower in query_lower:
                found_entities.append(entity_original)
        
        # If no entities found, try word matching
        if not found_entities:
            query_words = set(query_lower.split())
            for entity_lower, entity_original in self.entity_lower.items():
                entity_words = set(entity_lower.split())
                if entity_words & query_words:  # Intersection
                    found_entities.append(entity_original)
        
        return list(set(found_entities))  # Remove duplicates 