from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base
import json

class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Store JSON strings for elements and access_points
    elements_json = Column(Text, default="[]")
    access_points_json = Column(Text, default="[]")
    parameters_json = Column(Text, default="{}")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "elements": json.loads(self.elements_json),
            "access_points": json.loads(self.access_points_json),
            "parameters": json.loads(self.parameters_json)
        }
