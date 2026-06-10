# Knowledge Store 구조 점검 (G0)

- KS 디렉터리: `knowledge_store/dev/`
- 파일 수: **500** (이름 `{index}.json`, claim 위치 인덱스로 매핑)
- 총 레코드(JSONL 라인): **506640**
- 레코드 키: `['claim_id', 'query', 'type', 'url', 'url2text']`
- `type` 분포: `{'gold': 1096, 'question': 29380, 'question_duplicate': 79475, 'claim+question': 11871, 'answer': 27502, 'claim': 2362, 'gpt_question': 52908, 'background': 30708, 'background_questions': 47642, 'provenance': 37675, 'NER': 34255, 'most_similar': 49370, 'gpt_url_only': 60304, 'same_entity_questions': 42092}`
- `published_at` 필드 존재: **False** (없음 → 아카이브 스냅샷 타임스탬프로 유도)
- `source` 필드 존재: **False** (없음 → `url` netloc에서 도메인 추출)
- web.archive.org 스냅샷 비율: **0.1%** (350/506640, =published_at 유도 가능 비율)
- 아카이브 스냅샷 중 type=gold: **349/350** → 시점유도 가능 문서가 gold(검색 코퍼스에서 격리 대상)에 집중. 실제 검색 코퍼스의 published_at 커버리지는 사실상 0%.
- `url2text` passage 평균 길이: **750.5** (min 0, max 449035)
- claim_id ↔ 파일 인덱스 불일치: **0건** ✅ 전부 일치
