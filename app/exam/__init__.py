"""The IELTS Speaking mock test.

Live, during the test:
  ielts.py     - the fixed structure and timings, and the saved record (ExamLog)
  examiner.py  - everything the examiner says and asks
  engine.py    - ExamSession: walks the steps, records each answer

Afterwards:
  analysis.py  - fluency / vocabulary / grammar / pitch numbers from the recordings
  report.py    - band estimates and feedback -> report.json
  profile.py   - scores over time (data/profile.db)
"""
