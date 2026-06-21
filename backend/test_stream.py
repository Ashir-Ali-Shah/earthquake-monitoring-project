import sys
import time

def run_test():
    print("Importing simple_rag from main...")
    from main import simple_rag
    print("Testing query_stream...")
    
    question = "What were the strongest earthquakes in Japan this year?"
    print(f"Querying: {question}")
    
    try:
        gen = simple_rag.query_stream(question)
        for chunk in gen:
            print("Chunk:", chunk.strip())
    except Exception as e:
        print(f"Exception during stream: {e}")
        
    print("Stream finished.")
    print("Test complete.")

if __name__ == "__main__":
    run_test()
