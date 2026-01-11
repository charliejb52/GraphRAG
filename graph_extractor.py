import hashlib
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from connector import SECDataConnector

# Try to import OpenAI, fallback message if not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Install with: pip install openai")


class OperationalTriplet(BaseModel):
    """Pydantic model for operational relationships extracted from SEC filings."""
    subject_name: str = Field(..., description="Name of the subject entity (company)")
    relationship: str = Field(..., description="Relationship type: PRODUCES, DISTRIBUTES, HOLDS_ASSETS, or PROVIDES_SERVICES")
    object_name: str = Field(..., description="Name of the object entity")
    object_type: str = Field(..., description="Type: Public_Company, Private_Subsidiary, or Region")


def generate_deterministic_id(parent_cik: str, entity_name: str) -> str:
    """
    Generate a unique deterministic ID by hashing parent CIK and entity name.
    
    Args:
        parent_cik: The CIK of the parent company
        entity_name: The name of the entity (subsidiary)
    
    Returns:
        A unique hash string
    """
    combined = f"{parent_cik}:{entity_name}".lower().strip()
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


def extract_operational_relationships(text_chunk: str, parent_cik: str, api_key: Optional[str] = None) -> List[OperationalTriplet]:
    """
    Extract operational relationships from SEC 10-K text using an LLM.
    
    Args:
        text_chunk: A chunk of text from a 10-K filing
        parent_cik: The CIK of the parent company
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        List of OperationalTriplet objects
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is required. Install with: pip install openai")
    
    # Get API key from parameter or environment
    if api_key is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    # Few-shot system prompt
    system_prompt = """You are an expert at extracting operational relationships from SEC 10-K filings.

Extract relationships in the format:
- PRODUCES: Company produces/manufactures products or services
- DISTRIBUTES: Company distributes products to regions or entities
- HOLDS_ASSETS: Company holds assets in regions or subsidiaries
- PROVIDES_SERVICES: Company provides services to entities or regions

Object types:
- Public_Company: Publicly traded companies (have CIKs)
- Private_Subsidiary: Private subsidiaries or divisions
- Region: Geographic regions or locations

Return ONLY a valid JSON array of triplets (no markdown, no wrapper object). Each triplet must have: subject_name, relationship, object_name, object_type.

Example output:
[
  {"subject_name": "Apple Inc", "relationship": "PRODUCES", "object_name": "iPhone", "object_type": "Private_Subsidiary"},
  {"subject_name": "Apple Inc", "relationship": "DISTRIBUTES", "object_name": "Asia Pacific", "object_type": "Region"},
  {"subject_name": "Apple Inc", "relationship": "HOLDS_ASSETS", "object_name": "Ireland Operations", "object_type": "Private_Subsidiary"}
]"""

    user_prompt = f"""Extract all operational relationships from the following text from a 10-K filing:

{text_chunk}

Return a JSON array of triplets. The subject_name should be the parent company mentioned in the text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency, can change to gpt-4 if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic extraction
            response_format={"type": "json_object"} if False else None  # Some models support this
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON - sometimes the model returns JSON wrapped in markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
            # Handle if response is wrapped in a key
            if isinstance(data, dict) and len(data) == 1:
                data = list(data.values())[0]
            if isinstance(data, dict) and "triplets" in data:
                data = data["triplets"]
            elif isinstance(data, dict) and "relationships" in data:
                data = data["relationships"]
            
            # Ensure it's a list
            if not isinstance(data, list):
                data = [data] if isinstance(data, dict) else []
        except json.JSONDecodeError:
            # If direct parse fails, try to extract array from the response
            import re
            array_match = re.search(r'\[.*\]', content, re.DOTALL)
            if array_match:
                data = json.loads(array_match.group())
            else:
                print(f"Warning: Could not parse JSON from response: {content[:200]}")
                return []
        
        # Convert to OperationalTriplet objects
        triplets = []
        for item in data:
            try:
                triplet = OperationalTriplet(**item)
                triplets.append(triplet)
            except Exception as e:
                print(f"Warning: Skipping invalid triplet: {item}, Error: {e}")
                continue
        
        return triplets
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise


def ingest_triplets_to_neo4j(connector: SECDataConnector, triplets: List[OperationalTriplet], parent_cik: str):
    """
    Ingest operational triplets into Neo4j using MERGE to prevent duplicates.
    
    Args:
        connector: SECDataConnector instance
        triplets: List of OperationalTriplet objects to ingest
        parent_cik: The CIK of the parent company
    """
    with connector.graph_driver.session(database=connector.db_name) as session:
        for triplet in triplets:
            try:
                # MERGE subject node (parent company) - using CIK as unique identifier
                # MERGE object node based on type
                # Build query with dynamic relationship type
                relationship_type = triplet.relationship
                
                if triplet.object_type == "Private_Subsidiary":
                    # Use deterministic hash ID for subsidiaries
                    object_id = generate_deterministic_id(parent_cik, triplet.object_name)
                    query = f"""
                    MERGE (s:Company {{cik: $parent_cik}})
                    SET s.name = $subject_name
                    WITH s
                    MERGE (o:Subsidiary {{hash_id: $object_id}})
                    SET o.name = $object_name,
                        o.type = $object_type,
                        o.parent_cik = $parent_cik
                    WITH s, o
                    MERGE (s)-[r:{relationship_type}]->(o)
                    RETURN s, r, o
                    """
                    params = {
                        "parent_cik": parent_cik,
                        "subject_name": triplet.subject_name,
                        "object_id": object_id,
                        "object_name": triplet.object_name,
                        "object_type": triplet.object_type
                    }
                elif triplet.object_type == "Public_Company":
                    # For public companies, use name as identifier
                    query = f"""
                    MERGE (s:Company {{cik: $parent_cik}})
                    SET s.name = $subject_name
                    WITH s
                    MERGE (o:Company {{name: $object_name}})
                    SET o.type = $object_type
                    WITH s, o
                    MERGE (s)-[r:{relationship_type}]->(o)
                    RETURN s, r, o
                    """
                    params = {
                        "parent_cik": parent_cik,
                        "subject_name": triplet.subject_name,
                        "object_name": triplet.object_name,
                        "object_type": triplet.object_type
                    }
                else:  # Region
                    query = f"""
                    MERGE (s:Company {{cik: $parent_cik}})
                    SET s.name = $subject_name
                    WITH s
                    MERGE (o:Region {{name: $object_name}})
                    SET o.type = $object_type
                    WITH s, o
                    MERGE (s)-[r:{relationship_type}]->(o)
                    RETURN s, r, o
                    """
                    params = {
                        "parent_cik": parent_cik,
                        "subject_name": triplet.subject_name,
                        "object_name": triplet.object_name,
                        "object_type": triplet.object_type
                    }
                
                # Execute query - relationship type is already in the query string
                session.run(query, params)
                
            except Exception as e:
                print(f"Error ingesting triplet {triplet}: {e}")
                continue


# Test block
if __name__ == "__main__":
    # Sample paragraph from an SEC filing
    sample_text = """
    Apple Inc. manufactures its products through outsourcing partners, primarily located in Asia. 
    The Company has significant operations in China, where it produces the majority of its iPhone and iPad devices. 
    Apple distributes its products globally, with major distribution centers in Europe, Asia Pacific, and the Americas. 
    The Company holds substantial assets in Ireland through its subsidiary Apple Operations International. 
    Apple provides services through its AppleCare program to customers worldwide.
    """
    
    # Test CIK (Apple's CIK is 0000320193)
    test_cik = "0000320193"
    
    print("Testing graph extraction pipeline...")
    print("\n1. Testing deterministic ID generation...")
    test_id = generate_deterministic_id(test_cik, "Apple Operations International")
    print(f"   Generated ID for 'Apple Operations International': {test_id}")
    
    print("\n2. Testing relationship extraction (requires OpenAI API key)...")
    try:
        triplets = extract_operational_relationships(sample_text, test_cik)
        print(f"   Extracted {len(triplets)} triplets:")
        for i, triplet in enumerate(triplets, 1):
            print(f"   {i}. {triplet.subject_name} --[{triplet.relationship}]--> {triplet.object_name} ({triplet.object_type})")
        
        print("\n3. Testing Neo4j ingestion (requires Neo4j connection)...")
        try:
            connector = SECDataConnector()
            ingest_triplets_to_neo4j(connector, triplets, test_cik)
            print("   ✅ Successfully ingested triplets to Neo4j")
            connector.close()
        except Exception as e:
            print(f"   ❌ Neo4j ingestion failed: {e}")
            print("   (This is expected if Neo4j is not configured)")
    
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        print("   (This is expected if OpenAI API key is not set)")

