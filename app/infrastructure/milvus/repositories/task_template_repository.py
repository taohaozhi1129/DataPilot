from __future__ import annotations

from typing import Dict, List, Optional

from app.infrastructure.milvus.definitions import TASK_TEMPLATE_CATALOG
from app.infrastructure.milvus.repositories.base import BaseMilvusRepository


class TaskTemplateRepository(BaseMilvusRepository):
    def __init__(self, milvus_service=None) -> None:
        super().__init__(TASK_TEMPLATE_CATALOG, milvus_service=milvus_service)

    def search_templates(
        self,
        query_text: str,
        query_vector: List[float],
        *,
        limit: int = 3,
        tenant_id: Optional[str] = None,
        env: Optional[str] = None,
        template_type: Optional[str] = None,
        business_domain: Optional[str] = None,
    ) -> List[Dict]:
        filters = {
            "status": "ACTIVE",
            "is_active": True,
            "tenant_id": tenant_id,
            "env": env,
            "template_type": template_type,
            "business_domain": business_domain,
        }
        return self._search(query_text, query_vector, limit=limit, filters=filters)
