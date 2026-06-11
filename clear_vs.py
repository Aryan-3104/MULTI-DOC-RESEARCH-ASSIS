from src.embedder import cleanup_vectorstore_connections, delete_vectorstore\ncleanup_vectorstore_connections()\nok = delete_vectorstore()\nprint('delete_vectorstore returned', ok)
