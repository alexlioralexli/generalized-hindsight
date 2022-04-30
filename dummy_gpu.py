import torch
import subprocess
import time
import logging

# Takes about 8GB
ndim = 25_000
logging.basicConfig(format='[%(asctime)s] %(filename)s [%(levelname).1s] %(message)s', level=logging.DEBUG)

def get_gpu_usage():
    command = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    mem_total, mem_used, mem_free = map(lambda x: int(x), result.stdout.strip().split(","))
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
        if usage < 0.05:
            logging.debug("Running dummy GPU job for 30 seconds")
            run_dummy_job()
        else:
            logging.debug("Waiting for 30 seconds")
            time.sleep(30)

if __name__ == "__main__":
    main()
