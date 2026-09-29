import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm

load_dotenv(override=True)
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # Groq 배치(batch) API 클라이언트

MODEL = "openai/gpt-oss-20b"
BATCHES_FOLDER = "batches"  # 배치 입력(jsonl) 파일을 저장할 폴더명
OUTPUT_FOLDER = "output"  # 배치 처리 결과를 저장할 폴더명
state = Path("batches.pkl")  # 배치 목록 상태를 저장/복원할 pickle 파일 경로

# 이 프롬프트는 배치 API로 보내는 실제 요청 내용이므로 번역하지 않고 원문 그대로 유지
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    """상품 데이터를 정해진 크기(BATCH_SIZE)만큼 묶어 Groq 배치 API로 처리하는 클래스"""

    BATCH_SIZE = 1_000

    batches = []  # 생성된 모든 Batch 인스턴스를 담는 클래스 변수(리스트)

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None  # Groq에 업로드된 입력 파일의 id
        self.batch_id = None  # 제출된 배치 작업의 id
        self.output_file_id = None  # 완료된 배치의 결과 파일 id
        self.done = False
        folder = Path("lite") if lite else Path("full")  # lite 모드 여부에 따라 저장 폴더 분기
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        # 배치 API 규격에 맞는 한 줄짜리 JSON 요청(jsonl 포맷)을 만든다
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        # 이 배치에 속한 아이템들을 jsonl 파일로 저장
        batch_file = self.batches / self.filename
        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        # 만들어둔 jsonl 파일을 Groq에 업로드
        batch_file = self.batches / self.filename
        with batch_file.open("rb", encoding="utf-8") as f:
            response = groq.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        # 업로드한 파일로 배치 작업을 제출 (완료까지 최대 24시간 소요)
        response = groq.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        # 배치 작업 상태를 조회해서 완료 여부를 반환
        response = groq.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    def fetch_output(self):
        # 완료된 배치의 결과 파일을 다운로드
        output_file = str(self.output / self.filename)
        response = groq.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        # 다운로드한 결과를 읽어서 각 아이템의 summary(요약)에 반영
        output_file = str(self.output / self.filename)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        # 전체 items를 BATCH_SIZE 단위로 잘라 Batch 인스턴스들을 생성
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        # 모든 배치에 대해 파일 생성 -> 업로드 -> 제출을 순서대로 실행
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        # 아직 완료되지 않은 배치들의 완료 여부를 확인하고, 완료됐다면 결과를 가져와 반영
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        # items는 용량이 크므로 pickle에 저장하지 않고, 저장 전후로 잠시 떼어냈다 다시 붙인다
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb", encoding="utf-8") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        # 저장해둔 배치 상태를 복원하고, items를 다시 연결
        with state.open("rb", encoding="utf-8") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
