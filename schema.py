import uuid

from sqlalchemy import Column, Integer, String, DateTime, \
    Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///logging.db', echo=True)
Base = declarative_base()


class Trucks(Base):
    __tablename__ = 'trucks'
    id = Column(Integer, primary_key=True)
    ingestion_date = Column(DateTime)
    acquisition_date = Column(DateTime)
    original_filename = Column(String)
    image_uuid = Column(String, default=uuid.uuid4())
    license_plate = Column(String)
    valid_license_place = Column(Boolean)
    license_plate_bb = Column(String)
