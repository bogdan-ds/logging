from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///logtrucks.db', echo=True)
Base = declarative_base()


class Detections(Base):
    __tablename__ = "detections"
    id = Column(String, primary_key=True)
    ingestion_date = Column(DateTime)
    source_filename = Column(String)
    time_first_detected = Column(String, nullable=True)
    gdrive_file_id = Column(String, nullable=True)
    license_plate_prediction = Column(String, nullable=True)


Base.metadata.create_all(engine)
