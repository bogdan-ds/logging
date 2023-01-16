import uuid

from sqlalchemy import Column, Integer, String, DateTime, \
    Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///logtrucks.db', echo=True)
Base = declarative_base()


class Detections(Base):
    __tablename__ = 'trucks'
    id = Column(Integer, primary_key=True)
    ingestion_date = Column(DateTime)
    acquisition_date = Column(DateTime)
    source_filename = Column(String)
    gdrive_file_id = Column(String)
    image_uuid = Column(String, default=uuid.uuid4())
    license_plate_prediction = Column(String)
    time_first_detected = Column(String)
