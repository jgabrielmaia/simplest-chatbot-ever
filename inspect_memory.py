import chromadb
from rich.console import Console
from rich.table import Table


def inspect_memory(persist_dir: str = "./chroma_data"):
    """Display contents of ChromaDB conversation memory"""
    console = Console()
    
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            collection = client.get_collection("conversations")
        except:
            console.print("[yellow]No entries in memory store.[/yellow]")
            console.print("[dim]Start chatting with --memory to build conversation history.[/dim]")
            return
        
        results = collection.get(include=['documents', 'metadatas'])
        
        if not results['ids']:
            console.print("[yellow]No conversations stored yet. Start chatting with --memory to build memory.[/yellow]")
            return
        
        table = Table(title="Conversation Memory Store", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Session", style="green")
        table.add_column("Turn", style="yellow", justify="right")
        table.add_column("Content", style="white", max_width=70)
        table.add_column("Timestamp", style="blue")
        
        for doc_id, doc, meta in zip(results['ids'], results['documents'], results['metadatas']):
            content_preview = doc.replace('\n', ' ⏎ ')
            if len(content_preview) > 70:
                content_preview = content_preview[:67] + "..."
            
            table.add_row(
                doc_id,
                meta['session_id'][:8] + "...",
                str(meta['turn_number']),
                content_preview,
                meta['timestamp'].split('T')[0]  # Just date
            )
        
        console.print(table)
        console.print(f"\n[bold cyan]Total conversations:[/bold cyan] {len(results['ids'])}")
        
        sessions = set(meta['session_id'] for meta in results['metadatas'])
        console.print(f"[bold cyan]Unique sessions:[/bold cyan] {len(sessions)}")
        
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    inspect_memory()
