import timeit
import argparse
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings

file_log = "llm-mistral-rag.txt"

def log_and_print(msg, file_log=file_log):
    with open(file_log, "a+") as fp:
        fp.write(msg + "\n")
    print(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    parser.add_argument('--semantic_search',
                        type=bool,
                        default=False,
                        help='Enter True if you want to run semantic search, else False')
    args = parser.parse_args()

    log_and_print('\n' + '#'*60)
    query_msg = f'Query:\n {args.input}'
    log_and_print(query_msg)  
    if args.semantic_search:
        ts1 = timeit.default_timer()
        semantic_search = query_embeddings(args.input)
        ts2 = timeit.default_timer()
        log_and_print('='*50)
        log_and_print(f'Semantic search:\n {semantic_search}')
        log_and_print(f"\t [time] query_embeddings(): {(ts2-ts1):.2f}sec")
    else:
        ts1 = timeit.default_timer()
        qa_chain = setup_qa_chain()
        ts2 = timeit.default_timer()
        response = qa_chain({'query': args.input})
        ts3 = timeit.default_timer()
        log_and_print('=' * 50)
        log_and_print(f'Answer:\n {response["result"]}')
        log_and_print(f"\t [time] setup_qa_chain(): {(ts2-ts1):.2f}sec")
        log_and_print(f"\t [time] qa_chain(): {(ts3-ts2):.2f}sec")
