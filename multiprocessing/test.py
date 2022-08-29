from asyncio import as_completed
from re import S
import time
import multiprocessing
import concurrent.futures

def do_something(seconds):
    print(f"Sleeping for {seconds} second...")
    time.sleep(seconds)
    return f"Done sleeping...{seconds}"
    
start = time.perf_counter()

### Sequential ##3
# do_something()
# do_something()

### Multiprocessing ###
# p1 = multiprocessing.Process(target=do_something)
# p2 = multiprocessing.Process(target=do_something)

# p1.start()
# p2.start()

# # The process will finish before moving on in script
# p1.join()
# p2.join()

# processes = []
# for _ in range(10):
#     p = multiprocessing.Process(target=do_something, args=[1.5])
#     p.start()
#     processes.append(p)
    
# for process in processes:
#     process.join()

with concurrent.futures.ProcessPoolExecutor() as executor:
    # f1 = executor.submit(do_something, 1)
    # print(f1.result())
    
    # secs = [5, 4, 3, 2, 1]
    # results = [executor.submit(do_something, sec) for sec in secs]
    
    # for f in concurrent.futures.as_completed(results):
    #     print(f.result())
    
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)
    
    for result in results:
        print(result)
    

finish = time.perf_counter()

print(f"Finished in {round(finish-start, 2)} second(s)")
