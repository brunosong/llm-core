from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from pricer_ko.parser import parse
import os

CHUNK_SIZE = 1000

cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)  # CPU 코어 하나는 남겨두고 나머지를 워커로 사용


class ItemLoader:
    """HuggingFace 데이터셋을 받아 병렬로 정제해 Item 리스트로 만들어주는 로더"""

    def __init__(self, category):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        이 datapoint로부터 Item 생성을 시도한다
        성공하면 Item을, 포함시키면 안 되는 경우라면 None을 반환한다
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Dataset의 이 chunk(묶음)에 속한 요소들로부터 Item 리스트를 생성한다
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Dataset을 순회하면서 한 번에 CHUNK_SIZE만큼씩 datapoint 묶음을 만들어 넘겨준다
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        concurrent.futures를 사용해 datapoint 묶음 처리를 여러 프로세스에 분산시킨다 -
        처리 속도는 크게 빨라지지만, 실행되는 동안 컴퓨터 자원을 많이 사용하게 된다!
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS):
        """
        이 데이터셋을 불러온다; workers 매개변수는 데이터를 불러오고 정제하는 데
        사용할 프로세스 개수를 지정한다
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
