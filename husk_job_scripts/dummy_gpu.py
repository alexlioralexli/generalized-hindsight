import torch
import subprocess
import time
import logging
from subprocess import PIPE


"""
    ADAPTED FROM NIKHIL VERMA'S CODE

    PYTHON VERSION = 3.6

"""

# Takes about 8GB
ndim = 25_000
logging.basicConfig(format='[%(asctime)s] %(filename)s [%(levelname).1s] %(message)s', level=logging.DEBUG)

def get_gpu_usage():
    command = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), stdout=PIPE, stderr=PIPE)
    resultList = result.stdout.strip().split(b",")
    mem_total = resultList[0].decode("utf-8") 
    mem_used = resultList[1].decode("utf-8") 
    mem_free = resultList[2].decode("utf-8") 
    mem_used  = int(mem_used)
    mem_free = int(mem_free)
    logging.info(f"GPU Stats: Total: {mem_total}, Free: {mem_free} Used: {mem_used}")
    return mem_used / mem_free

def run_dummy_job():
    start = time.time()
    random1 = torch.randn([ndim, ndim]).to("cuda")
    random2 = torch.randn([ndim, ndim]).to("cuda")
    while time.time() - start < 0.5 * 60:
        random1 = random1 * random2
        random2 = random2 * random1
    del random1, random2
    torch.cuda.empty_cache()

def main():
    while True:
        usage = get_gpu_usage()
        if usage < 0.2:
            logging.debug("Running dummy GPU job for 30 seconds")
            run_dummy_job()
        else:
            logging.debug("Waiting for 30 seconds")
            time.sleep(30)

if __name__ == "__main__":
    main()
