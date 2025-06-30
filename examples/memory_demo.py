"""
Demonstration of the Persistent Memory System.

This script shows how to use the memory system through the API.
"""
import asyncio
import aiohttp

class MemoryDemo:
    """
    MemoryDemo class for demonstrating the memory system.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000", user_id: str = "demo_user"):
        self.base_url = base_url
        self.user_id = user_id
        self.headers = {"X-User-ID": user_id}

    async def demo(self):
        """Run the complete demonstration."""
        async with aiohttp.ClientSession() as session:
            print("🧠 GraphRAG Persistent Memory System Demo\n")

            # 1. Set up user
            print("1️⃣ Setting up user profile...")
            await self.setup_user(session)

            # 2. Create contexts
            print("\n2️⃣ Creating contexts...")
            await self.create_contexts(session)

            # 3. Process queries with memory
            print("\n3️⃣ Processing queries with memory...")
            await self.process_queries(session)

            # 4. Search memory
            print("\n4️⃣ Searching through memory...")
            await self.search_memory(session)

            # 5. Show statistics
            print("\n5️⃣ Memory statistics...")
            await self.show_statistics(session)

            # 6. List conversations
            print("\n6️⃣ Listing conversations...")
            await self.list_conversations(session)

    async def setup_user(self, session: aiohttp.ClientSession):
        """Set up user profile."""
        data = {
            "user_id": self.user_id,
            "name": "Demo User",
            "email": "demo@example.com"
        }

        async with session.post(f"{self.base_url}/users/set", json=data) as resp:
            result = await resp.json()
            print(f"✅ User created: {result['user_id']} ({result['name']})")

    async def create_contexts(self, session: aiohttp.ClientSession):
        """Create different contexts."""
        contexts = [
            {
                "name": "Machine Learning Research",
                "description": "Research on ML algorithms and papers",
                "set_as_active": True
            },
            {
                "name": "Web Development Project",
                "description": "Frontend and backend development tasks",
                "set_as_active": False
            }
        ]

        for ctx in contexts:
            async with session.post(
                f"{self.base_url}/contexts",
                json=ctx,
                headers=self.headers
            ) as resp:
                result = await resp.json()
                print(f"✅ Created context: {result['name']} (ID: {result['id']})")

    async def process_queries(self, session: aiohttp.ClientSession):
        """Process various queries to build memory."""
        queries = [
            {
                "query": "What are the key differences between supervised and unsupervised learning?",
                "agent_name": None  # Let system choose
            },
            {
                "query": "Can you explain gradient descent optimization?",
                "agent_name": None
            },
            {
                "query": "Summarize the latest trends in deep learning",
                "agent_name": "summary"
            }
        ]

        for i, query_data in enumerate(queries, 1):
            print(f"\n📝 Query {i}: {query_data['query'][:50]}...")

            async with session.post(
                f"{self.base_url}/query",
                json=query_data,
                headers=self.headers
            ) as resp:
                result = await resp.json()
                print(f"✅ Processed by agents: {', '.join(result['agents_used'])}")

                # Show first part of response
                for agent, response in result['responses'].items():
                    if isinstance(response, dict) and 'response' in response:
                        preview = response['response'][:100] + "..."
                        print(f"   {agent}: {preview}")

    async def search_memory(self, session: aiohttp.ClientSession):
        """Search through conversation memory."""
        search_queries = [
            "gradient descent",
            "learning algorithms",
            "optimization"
        ]

        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")

            data = {
                "query": query,
                "limit": 3
            }

            async with session.post(
                f"{self.base_url}/memory/search",
                json=data,
                headers=self.headers
            ) as resp:
                result = await resp.json()
                print(f"Found {result['total_results']} results:")

                for res in result['results'][:3]:
                    print(f"  - {res['role']}: {res['content'][:60]}...")
                    print(f"    Relevance: {res['relevance_score']:.2f}")

    async def show_statistics(self, session: aiohttp.ClientSession):
        """Show memory usage statistics."""
        async with session.get(
            f"{self.base_url}/memory/statistics",
            headers=self.headers
        ) as resp:
            stats = await resp.json()

            print("\n📊 Memory Statistics:")
            print(f"  Total conversations: {stats.get('total_conversations', 0)}")
            print(f"  Total messages: {stats.get('total_messages', 0)}")
            print(f"  Total contexts: {stats.get('total_contexts', 0)}")

            if 'agent_usage' in stats:
                print("\n  Agent usage:")
                for agent, count in stats['agent_usage'].items():
                    print(f"    - {agent}: {count} messages")

    async def list_conversations(self, session: aiohttp.ClientSession):
        """List recent conversations."""
        async with session.get(
            f"{self.base_url}/conversations",
            headers=self.headers,
            params={"limit": 5}
        ) as resp:
            conversations = await resp.json()

            print(f"\n💬 Recent Conversations ({len(conversations)} total):")
            for conv in conversations:
                print(f"  - {conv['title']}")
                print(f"    Messages: {conv['message_count']}, Created: {conv['created_at'][:10]}")

                # Get conversation details
                async with session.get(
                    f"{self.base_url}/conversations/{conv['id']}",
                    headers=self.headers
                ) as detail_resp:
                    details = await detail_resp.json()
                    print("    Last messages:")
                    for msg in details['messages'][-2:]:
                        preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                        print(f"      {msg['role']}: {preview}")


async def main():
    """Run the demonstration."""
    demo = MemoryDemo()

    print("Starting Memory System Demo...")
    print("Make sure the API server is running on http://127.0.0.1:8000")
    print("-" * 60)

    try:
        await demo.demo()
        print("\n✅ Demo completed successfully!")
    except aiohttp.ClientError as e:
        print(f"\n❌ Error: Could not connect to API server: {e}")
        print("Make sure the server is running with: python -m src.api.main")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
