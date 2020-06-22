import time
import multiprocessing as mp

def pow_delay(q, data):
    for x in data:
        print(q, x)
        q.put(x * x)
        time.sleep(x)

if __name__ == '__main__':
    queue = mp.Queue()
    pool = mp.Pool(processes=4)

    pool.apply_async(pow_delay, args=(queue, list(range(10))))
    
    while True:
        print(queue.get())
    input('wait')