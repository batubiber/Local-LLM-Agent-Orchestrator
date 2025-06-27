"""
Milvus-based vector store for GraphRAG system.
"""
from __future__ import annotations

from typing import List, Optional, Dict
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class MilvusGraphStore:
    """Milvus-based vector store for GraphRAG with entity and relationship collections."""
    
    def __init__(
        self,
        uri: str,
        token: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize Milvus GraphRAG store.
        
        Args:
            uri: Milvus connection URI
            token: Authentication token for Milvus
            embedding_model: Sentence transformer model name
            embedding_dim: Embedding dimension (auto-detected if None)
        """
        self.client = MilvusClient(uri=uri, token=token)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Auto-detect embedding dimension if not provided
        if embedding_dim is None:
            self.embedding_dim = len(self.embedding_model.encode("test"))
        else:
            self.embedding_dim = embedding_dim
            
        # Collection names
        self.entity_collection = "entity_collection"
        self.relation_collection = "relation_collection"
        self.passage_collection = "passage_collection"
        
        # Adjacency mappings
        self.entityid_2_relationids = defaultdict(list)
        self.relationid_2_passageids = defaultdict(list)
        
        # Create collections
        self._create_collections()
    
    def _create_collections(self) -> None:
        """Create Milvus collections for entities, relations, and passages."""
        collections = [
            self.entity_collection,
            self.relation_collection, 
            self.passage_collection
        ]
        
        for collection_name in collections:
            try:
                if self.client.has_collection(collection_name=collection_name):
                    print(f"Collection {collection_name} already exists, dropping...")
                    self.client.drop_collection(collection_name=collection_name)
                
                print(f"Creating collection {collection_name}...")
                self.client.create_collection(
                    collection_name=collection_name,
                    dimension=self.embedding_dim,
                    consistency_level="Strong",
                )
                print(f"✅ Created collection {collection_name}")
            except Exception as e:
                print(f"Error creating collection {collection_name}: {e}")
                raise
    
    def insert_data(
        self,
        entities: List[str],
        relations: List[str], 
        passages: List[str],
        entityid_2_relationids: Dict[int, List[int]],
        relationid_2_passageids: Dict[int, List[int]]
    ) -> None:
        """
        Insert data into Milvus collections.
        
        Args:
            entities: List of entity texts
            relations: List of relation texts
            passages: List of passage texts
            entityid_2_relationids: Mapping from entity IDs to relation IDs
            relationid_2_passageids: Mapping from relation IDs to passage IDs
        """
        try:
            # Store adjacency mappings
            self.entityid_2_relationids = entityid_2_relationids
            self.relationid_2_passageids = relationid_2_passageids
            
            # Insert entities
            if entities:
                print(f"Inserting {len(entities)} entities...")
                self._insert_collection(self.entity_collection, entities)
                print("✅ Entities inserted successfully")
            else:
                print("⚠️ No entities to insert")
                
            # Insert relations
            if relations:
                print(f"Inserting {len(relations)} relations...")
                self._insert_collection(self.relation_collection, relations)
                print("✅ Relations inserted successfully")
            else:
                print("⚠️ No relations to insert")
                
            # Insert passages
            if passages:
                print(f"Inserting {len(passages)} passages...")
                self._insert_collection(self.passage_collection, passages)
                print("✅ Passages inserted successfully")
            else:
                print("⚠️ No passages to insert")
                
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise
    
    def _insert_collection(self, collection_name: str, texts: List[str]) -> None:
        """Insert texts into a specific collection."""
        batch_size = 512
        
        for row_id in range(0, len(texts), batch_size):
            batch_texts = texts[row_id:row_id + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            
            batch_ids = [row_id + j for j in range(len(batch_texts))]
            batch_data = [
                {
                    "id": id_,
                    "text": text,
                    "vector": vector.tolist(),
                }
                for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
            ]
            
            self.client.insert(
                collection_name=collection_name,
                data=batch_data,
            )
    
    def search_entities(self, query_texts: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Search for similar entities, batching requests to respect Milvus nq limit (max 10)."""
        query_embeddings = [
            self.embedding_model.encode(query_text).tolist() 
            for query_text in query_texts
        ]
        #########################################################
        # This code below is the original code, but it is not working because of the nq limit of Milvus.
        # So we are using the batching approach to respect the nq limit.
        # results = self.client.search(
        #     collection_name=self.entity_collection,
        #     data=query_embeddings,
        #     limit=top_k,
        #     output_fields=["id", "text"],
        # )
        # return results
        #########################################################
        
        BATCH_SIZE = 10  # Milvus nq limit
        all_results = []
        for i in range(0, len(query_embeddings), BATCH_SIZE):
            batch_embeddings = query_embeddings[i:i+BATCH_SIZE]
            batch_results = self.client.search(
                collection_name=self.entity_collection,
                data=batch_embeddings,
                limit=top_k,
                output_fields=["id", "text"],
            )
            all_results.extend(batch_results)
        return all_results
    
    def search_relations(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar relations."""
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.client.search(
            collection_name=self.relation_collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "text"],
        )[0]
        
        return results
    
    def search_passages(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar passages (baseline comparison)."""
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.client.search(
            collection_name=self.passage_collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "text"],
        )[0]
        
        return results
    
    def get_entity_relations(self, entity_ids: List[int]) -> List[int]:
        """Get all relation IDs connected to given entity IDs."""
        relation_ids = set()
        for entity_id in entity_ids:
            relation_ids.update(self.entityid_2_relationids.get(entity_id, []))
        return list(relation_ids)
    
    def get_relation_passages(self, relation_ids: List[int]) -> List[int]:
        """Get all passage IDs connected to given relation IDs."""
        passage_ids = set()
        for relation_id in relation_ids:
            passage_ids.update(self.relationid_2_passageids.get(relation_id, []))
        return list(passage_ids)
    
    def get_passages_by_ids(self, passage_ids: List[int]) -> List[str]:
        """Retrieve passage texts by their IDs."""
        if not passage_ids:
            return []
            
        results = self.client.query(
            collection_name=self.passage_collection,
            filter=f"id in {passage_ids}",
            output_fields=["id", "text"],
        )
        
        # Sort by original order
        id_to_text = {res["id"]: res["text"] for res in results}
        return [id_to_text.get(pid, "") for pid in passage_ids if pid in id_to_text]
    
    def get_relations_by_ids(self, relation_ids: List[int]) -> List[str]:
        """Retrieve relation texts by their IDs."""
        if not relation_ids:
            return []
            
        results = self.client.query(
            collection_name=self.relation_collection,
            filter=f"id in {relation_ids}",
            output_fields=["id", "text"],
        )
        
        # Sort by original order
        id_to_text = {res["id"]: res["text"] for res in results}
        return [id_to_text.get(rid, "") for rid in relation_ids if rid in id_to_text] 