from sqlmodel import Session, select, func
from app.db.database import engine
from app.models.user import User
from app.models.job import Job
from app.models.transcript import Transcript
from app.models.summary import Summary

def check_db():
    with Session(engine) as session:
        users = session.exec(select(func.count(User.id))).one()
        jobs = session.exec(select(func.count(Job.id))).one()
        transcripts = session.exec(select(func.count(Transcript.id))).one()
        summaries = session.exec(select(func.count(Summary.id))).one()

        print("--- THỐNG KÊ DỮ LIỆU TRONG SUPABASE ---")
        print(f"Số lượng Tài khoản (User): {users}")
        print(f"Số lượng File âm thanh (Job): {jobs}")
        print(f"Số lượng Câu thoại (Transcript): {transcripts}")
        print(f"Số lượng Bản tóm tắt (Summary): {summaries}")
        print("-----------------------------------------")

if __name__ == "__main__":
    check_db()
