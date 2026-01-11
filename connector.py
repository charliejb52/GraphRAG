import os
import chromadb
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

class SECDataConnector:
    def __init__(self):
        # ChromaDB initialization with Client
        self.vector_client = chromadb.Client()
        self.vector_collection = self.vector_client.get_or_create_collection(name="sec_8k_filings")
        
        # Neo4j Aura Connection
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not neo4j_uri or not neo4j_username or not neo4j_password:
            raise ValueError(
                "Neo4j environment variables not set. "
                "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file"
            )
        
        self.graph_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        self.db_name = os.getenv("NEO4J_DATABASE", "neo4j")

    def verify_connections(self):
        # Check Neo4j
        try:
            self.graph_driver.verify_connectivity()
            print("✅ Neo4j Connection Successful")
        except Exception as e:
            print(f"❌ Neo4j Connection Failed: {e}")

        # Check ChromaDB
        print(f"✅ ChromaDB Connection Successful (Collection: {self.vector_collection.name})")

    def close(self):
        if self.graph_driver is not None:
            self.graph_driver.close()

# Execute verification if script is run directly
if __name__ == "__main__":
    connector = SECDataConnector()
    connector.verify_connections()
    connector.close()
