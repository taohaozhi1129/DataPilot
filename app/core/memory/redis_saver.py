from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple, SerializerProtocol
import redis.asyncio as redis
import json

class AsyncRedisSaver(BaseCheckpointSaver):
    """
    Redis-based CheckpointSaver for LangGraph.
    Stores checkpoints in Redis hashes.
    Key format: "checkpoint:{thread_id}:{checkpoint_id}"
    """
    
    def __init__(
        self, 
        client: redis.Redis, 
        serde: Optional[SerializerProtocol] = None
    ):
        super().__init__(serde=serde)
        self.client = client

    @classmethod
    def from_url(cls, url: str, **kwargs):
        client = redis.from_url(url, **kwargs)
        return cls(client)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis."""
        thread_id = config["configurable"]["thread_id"]
        # Find the latest checkpoint for this thread if not specified
        # In a full implementation, we would maintain a sorted set or list of checkpoints per thread
        # For simplicity, we'll assume we store the "latest" pointer in "thread:{thread_id}:latest"
        
        if "checkpoint_id" in config["configurable"]:
            checkpoint_id = config["configurable"]["checkpoint_id"]
        else:
            checkpoint_id = await self.client.get(f"thread:{thread_id}:latest")
            if not checkpoint_id:
                return None
            checkpoint_id = checkpoint_id.decode("utf-8")

        key = f"checkpoint:{thread_id}:{checkpoint_id}"
        data = await self.client.hgetall(key)
        
        if not data:
            return None
            
        # Decode fields
        checkpoint_bytes = data.get(b"checkpoint")
        checkpoint_type = data.get(b"checkpoint_type")
        metadata_bytes = data.get(b"metadata")
        metadata_type = data.get(b"metadata_type")
        parent_checkpoint_id = data.get(b"parent_checkpoint_id")
        
        if not checkpoint_bytes:
            return None

        # Deserialize
        # Handle typed serialization (LangGraph >= 0.2.x style)
        if hasattr(self.serde, "loads_typed") and checkpoint_type:
            checkpoint = self.serde.loads_typed((checkpoint_type.decode(), checkpoint_bytes))
            metadata = self.serde.loads_typed((metadata_type.decode(), metadata_bytes)) if metadata_bytes and metadata_type else {}
        else:
            # Fallback for older serializers or missing type info (assume simple load)
            # Note: JsonPlusSerializer does NOT have loads(), so this path handles cases where serde is something else
            # or we have old data but new serializer (which might fail if we don't have type).
            # For this fix, we assume we are fixing for JsonPlusSerializer.
            if hasattr(self.serde, "loads"):
                 checkpoint = self.serde.loads(checkpoint_bytes)
                 metadata = self.serde.loads(metadata_bytes) if metadata_bytes else {}
            else:
                 # If we have bytes but no type and no loads method, we can't deserialize safely.
                 # But we might try to guess or just fail. 
                 # Given the error, we MUST use loads_typed. 
                 # If checkpoint_type is missing (old data), we might default to "msgpack" if we knew it was msgpack.
                 # But here we will assume data is written with new format or we accept failure on old data.
                 # Let's try to default to "msgpack" if type is missing but we have JsonPlusSerializer.
                 checkpoint = self.serde.loads_typed(("msgpack", checkpoint_bytes))
                 metadata = self.serde.loads_typed(("msgpack", metadata_bytes)) if metadata_bytes else {}

        parent_id = parent_checkpoint_id.decode("utf-8") if parent_checkpoint_id else None

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config={"configurable": {"thread_id": thread_id, "checkpoint_id": parent_id}} if parent_id else None
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints. (Simplified implementation)"""
        # Full implementation would require scanning keys or using sorted sets.
        # This is a placeholder for basic functionality.
        # For production, use a Sorted Set (ZSET) to store checkpoint_ids by timestamp.
        yield  # Empty iterator for now

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Save a checkpoint to Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        key = f"checkpoint:{thread_id}:{checkpoint_id}"
        latest_key = f"thread:{thread_id}:latest"
        
        # Serialize
        if hasattr(self.serde, "dumps_typed"):
            cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
            md_type, md_bytes = self.serde.dumps_typed(metadata)
            checkpoint_data = {
                "checkpoint": cp_bytes,
                "checkpoint_type": cp_type,
                "metadata": md_bytes,
                "metadata_type": md_type,
            }
        else:
            checkpoint_data = {
                "checkpoint": self.serde.dumps(checkpoint),
                "metadata": self.serde.dumps(metadata),
            }
            
        if parent_checkpoint_id:
            checkpoint_data["parent_checkpoint_id"] = parent_checkpoint_id

        # Use transaction (pipeline)
        async with self.client.pipeline(transaction=True) as pipe:
            # Store the checkpoint data
            await pipe.hset(key, mapping=checkpoint_data)
            # Set expiration (e.g., 7 days)
            await pipe.expire(key, 60 * 60 * 24 * 7)
            
            # Update latest pointer
            await pipe.set(latest_key, checkpoint_id)
            await pipe.expire(latest_key, 60 * 60 * 24 * 7)
            
            await pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes to Redis.
        
        This is a simplified implementation that acknowledges the writes but does not
        persist them for resumption. For full production support allowing resumption
        from interruptions, these writes should be stored in Redis.
        """
        pass
